module Redbird

redbird_m_path = Base.Filesystem.joinpath(@__DIR__(), "redbird-m/matlab/")

include("iso2mesh.jl")
include("structs.jl")
include("forward.jl")

end # module Redbird
