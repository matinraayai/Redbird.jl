using SparseArrays
using Zygote
using LinearSolve
using ChainRulesCore
using Zygote
using ForwardDiff
using Krylov
using ChainRulesTestUtils
using FillArrays


function linearSolve(A::AbstractMatrix{T}, B::AbstractVector{T}) where {T <: Number}
    qmr(A, B)[1]
end

function ChainRulesCore.rrule(::typeof(linearSolve), 
               A::AbstractMatrix{T}, 
               B::AbstractVector{T}) where {T <: Number}
    u = qmr(A, B)
    project_B = ProjectTo(B)
    function pullback(ū)
        Ū = unthunk(ū)
        projected_Ū = project_B(Array(Ū))
        # if projected_Ū isa Fill
        #     projected_Ū = Array(projected_Ū)
        # end
        z = qmr(A', projected_Ū)[1]
        @show Array(A')
        @show z, Array(projected_Ū)
        @show u[1]
        dA = @thunk(-z * u[1]')
        dB = z
        return (NoTangent(), dA, dB)
    end
    return u[1], pullback
end

function main()
    
    Zygote.refresh()
    Amat = sprand(Float64, 10, 10, 0.8)
    B = sprand(Float64, 10, 0.8)
    Amat_dense = Array(Amat)
    B_dense = Array(B)
    # Amat = rand(20, 20)
    # B = rand(20)
    test_rrule(linearSolve, Amat_dense, B_dense, atol=1e-5, rtol=1e-5)
    # test_rrule(linearSolve, Amat, B, atol=1e-5, rtol=1e-5)
    # (res_dense, ) = gradient(x -> sum(linearSolve(x, B_dense)), Amat_dense)
    # (res_sparse, ) = gradient(x -> sum(linearSolve(x, B)), Amat)
    # @show res_dense, Array(res_sparse)

end

main()