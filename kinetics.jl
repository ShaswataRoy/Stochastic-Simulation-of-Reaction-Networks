using Distributions
using DataFrames
using StaticArrays
using Random
using BenchmarkTools
using StatsBase
using Plots
using Pipe
using Optim
using FastExpm
using TensorCast
using CUDA
using CUDA.CUSPARSE
using SparseArrays
using LinearAlgebra
using JLD

struct SSAStats
    termination_status::String
    nsteps::Int64
end

struct SSAArgs{X,Ftype,N,P}
    x0::X
    F::Ftype
    nu::N
    parms::P
    tf::Float64
    alg::Symbol
    tvc::Bool
end

struct SSAResult
    time::Vector{Float64}
    data::Matrix{Int64}
    stats::SSAStats
    args::SSAArgs
end

function pfsample(w::AbstractArray{Float64,1},s::Float64,n::Int64)
    t = rand() * s
    i = 1
    cw = w[1]
    while cw < t && i < n
        i += 1
        @inbounds cw += w[i]
    end
    return i
end

function gillespie(x0::AbstractVector{Int64},F::Base.Callable,nu::AbstractMatrix{Int64},
    parms::AbstractVector{Float64},tf::Float64)
    # Args
    args = SSAArgs(x0,F,nu,parms,tf,:gillespie,false)
    # Set up time array
    ta = Vector{Float64}()
    t = 0.0
    push!(ta,t)
    # Set up initial x
    nstates = length(x0)
    x = copy(x0')
    xa = copy(Array(x0))
    # Number of propensity functions
    numpf = size(nu,1)
    # Main loop
    termination_status = "finaltime"
    nsteps = 0
    while t <= tf
        pf = F(x,parms)
        # Update time
        sumpf = sum(pf)
        if sumpf == 0.0
            termination_status = "zeroprop"
            break
        end
        dt = rand(Exponential(1/sumpf))
        t += dt
        push!(ta,t)
        # Update event
        ev = pfsample(pf,sumpf,numpf)
        if x isa SVector
            @inbounds x[1] += nu[ev,:]
        else
            deltax = view(nu,ev,:)
            for i in 1:nstates
                @inbounds x[1,i] += deltax[i]
            end
        end
        for xx in x
            push!(xa,xx)
        end
        # update nsteps
        nsteps += 1
    end
    stats = SSAStats(termination_status,nsteps)
    xar = transpose(reshape(xa,length(x),nsteps+1))
    return SSAResult(ta,xar,stats,args)
end

function ssa(x₀::AbstractVector{Int64},F::Base.Callable,nu::AbstractMatrix{Int64},parms::AbstractVector{Float64},
    times::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64})
    tf = times[end]
    
    gillespie_sim = gillespie(x₀,F,nu,parms,tf)
    ssa_result = Array{Int64,2}(undef,length(times),length(x₀))

    event_indx = 1
    for (time_indx,time) in enumerate(times)
        while gillespie_sim.time[event_indx]<time
            event_indx += 1
        end
        ssa_result[time_indx,:] = gillespie_sim.data[event_indx,:]
    end

    return ssa_result
end

function F(x,parms)
    (DNA_Inactive,DNA_Active,mRNA,Protein) = x
    (kᴰᴺᴬ₁,kᴰᴺᴬ₂,kᵐ₁,kᵐ₂,kᴾ₁,kᴾ₂,r) = parms
    switch_Active = kᴰᴺᴬ₁*DNA_Inactive
    switch_Inactive = kᴰᴺᴬ₂*DNA_Active
    mRNA_birth = kᵐ₁*DNA_Active
    mRNA_death = kᵐ₂*mRNA
    protein_birth = kᴾ₁*mRNA
    protein_death = kᴾ₂*Protein
    repression =  r*Protein*DNA_Active
    return [switch_Active,switch_Inactive+repression,
    mRNA_birth,mRNA_death,
    protein_birth,protein_death]
  end


x₀ = [0,1,0,0]
ν = [[-1 1 0 0];[1 -1 0 0];[0 0 1 0];[0 0 -1 0];[0 0 0 1];[0 0 0 -1];[1 -1 0 0]]
reaction_rates = [1.2,0.5,3.,.5,3.,0.5,.2]/2
tf = 10.
Random.seed!(1234)
  
result = gillespie(x₀,F,ν,reaction_rates,tf)

times = 0:.01:50
n_simulations = 1000
species = Array{Int64,3}(undef,n_simulations,length(times),length(x₀))

for i in range(1,n_simulations)
    species[i,:,:] = ssa(x₀,F,ν,reaction_rates,times)
end

@cast data[(i,j),k] := species[i,j,k];

state_space = unique(data,dims=1);

init_prob = zeros(Int64,size(state_space,1));
init_prob[1] = 1;

mRNA_max = state_space[:,3] |> maximum


function expCU(A_gpu::CuArray{Float64,2};threshold=1e-6)
    rows = LinearAlgebra.checksquare(A_gpu);
    P = zeros(rows,rows);
    P[diagind(P)] = ones(rows);
    # P_gpu = CuSparseMatrixCSR(P);
    P_gpu = CuArray{Float64,2}(P);

    next_term = copy(P_gpu);
    n = 1;
    delta=norm(next_term,Inf);
    # nonzero_tol = 1e-12

    while delta>threshold
        next_term=(1/n)*A_gpu*next_term;
        # next_term=droptolerance!(next_term, nonzero_tol);
        
        delta=norm(next_term,Inf);
        P_gpu = P_gpu.+ next_term; n=n+1;
    end

    CUDA.reclaim()
    return P_gpu
end

function index_reaction(state_change::AbstractVector{Int64},updates::AbstractMatrix{Int64})
    for (indx,update) in enumerate(eachrow(updates))
        if update==state_change
            return indx
        end
    end 
    return -1
end

function droptolerance!(A::CuSparseMatrixCSR{Float64, Int32}, tolerance)
    A .= tolerance*round.((1/tolerance).*A)
end

function state_distribution(t::Float64,reaction_rates::AbstractArray{Float64,1},
    states::AbstractArray{Int64,2},updates::AbstractArray{Int64,2},init_prob::AbstractArray{Int64,1})
    
    n = size(states,1)
    Q_Matrix = Matrix{Float64}(undef,n,n)

    for i in range(1,n)
        for j in range(1,n)
            v = states[j,:]-states[i,:]

            index = index_reaction(v,updates)
            if index>=0
                propensity = F(states[i,:],reaction_rates)
                Q_Matrix[j,i] = propensity[index]
            else
                Q_Matrix[j,i] = 0
            end
        end
    end

    for i in range(1,n)
        Q_Matrix[i,i] = -sum(Q_Matrix[:,i])
    end

    save("data.jld", "data", Q_Matrix)

    Q_gpu = CuArray{Float64,2}(Q_Matrix)
    @time exp_Q = expCU(Q_gpu*t; threshold=1e-6) |> Matrix
    
    state_distribution = exp_Q*init_prob

    return state_distribution
end

function count_distribution(t::Float64,reaction_rate::AbstractArray{Float64,1},
    states::AbstractArray{Int64,2},updates::AbstractArray{Int64,2},init_prob::AbstractArray{Int64,1})

    count_distribution = Vector{Vector{Float64}}(undef, size(states,2))
    state_distribution_array = state_distribution(t,reaction_rate,states,updates,init_prob)

    for (state_indx,state) in enumerate(eachcol(states))
        count_distribution[state_indx] = zeros(length(unique!(sort!(vcat(state)))))
        for (indx,state_inst) in enumerate(eachrow(states))
            count_distribution[state_indx][state_inst[state_indx]+1] += state_distribution_array[indx]
        end
    end
    
    return count_distribution
end

function likelihood(reaction_rates::AbstractArray{Float64,1},data::AbstractArray{Float64,2},
    states::AbstractArray{Int64,2},updates::AbstractArray{Int64,2},init_prob::AbstractArray{Int64,1})
    likelihood = 0
    for j in range(1,6)
        mRNA_count_dist = count_distribution(2.0*j,reaction_rates,states,updates,init_prob)[3][1:mRNA_max]
        likelihood += data[j,:]'*log.(mRNA_count_dist)
    end
    return likelihood
end

data_hist = zeros(6,mRNA_max)

for (indx,specie) in enumerate(eachcol(species[:,200:200:1200,3]))
    count = specie |> countmap 
    prob = @pipe count  |> sort(_,by=x->x[1]) |> values |> collect |> _/sum(_)
    data_hist[indx,1:length(prob)] = prob
end

@time count_distribution(2.0,reaction_rates,state_space,ν,init_prob)
# @time print(likelihood(reaction_rates,data_hist,state_space,ν,init_prob))