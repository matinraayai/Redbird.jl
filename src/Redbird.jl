module Redbird

redbird_m_path = Base.Filesystem.joinpath(@__DIR__(), "redbird-m/matlab/")

include("structs.jl")
include("jlmat.jl")
include("iso2mesh.jl")
include("forward.jl")

end # module Redbird
