using Flux, CSV
using CUDA
using  Flux.Data: DataLoader
using Flux: @epochs, flatten
using Flux.Losses: logitbinarycrossentropy
using Statistics
using DataFrames
using BSON: @save, @load
using BSON
using Zygote

pwd()
cd("GoL")
#Data wrangling
train_data = CSV.File("train.csv"; header =1, drop=[:id, :delta]);
test_data = CSV.File("test.csv", drop=[:id, :delta]);
function squarify(data::CSV.File{true}) 
    rows = [[i for i in data[j]] for j in 1:length(data)]
    starts = Array{Float32}(undef,25, 25,1, length(data))
    stops = Array{Float32}(undef,25, 25,1, length(data))
    for (i,row) in enumerate(rows)
        two_squares = permutedims(reshape(row,25,25,1,2),(2,1,3,4))
        starts[:,:,:,i] = two_squares[:,:,:,1]
        stops[:,:,:,i] = two_squares[:,:,:,2]
    end
    
    return(starts, stops)
end

y, x = squarify(train_data)  

#Loading data
y_train = y[:,:,:,1:40000]
x_train = x[:,:,:,1:40000]
y_verify = y[:,:,:,40001:end]
x_verify = x[:,:,:,40001:end]
data_train = DataLoader(x_train, y_train, batchsize = 200, shuffle = true)
verify = DataLoader(x_verify, y_verify, batchsize = 1000 )

##creating model
model = Chain( 
    Conv((3,3),1=>24, pad=(1,1), elu),
    BatchNorm(24),
    Dropout(0.4),
    Conv((3,3),24=>64, pad=(1,1),elu),
    BatchNorm(64),
    Dropout(0.4),
    Conv((3,3),64=>128, pad=(1,1),elu),
    BatchNorm(128),
    Dropout(0.4),
    Conv((3,3),128=>1, pad=(1,1), sigmoid),
)


loss(x,y) = logitbinarycrossentropy(model(x),y)


function  lossmean(data_loader, model)
    l = 0
    for (x,y) in data_loader
        l += logitbinarycrossentropy(model(x), y)
    end
    return l/length(data_loader)
end


function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        values = cpu(model(x))
        final = [i[1] > 0.5 ? 1 : 0 for i in values]
        acc += sum(final .== cpu(y))/prod(size(x))
    end
    return acc/length(data_loader)
end
@load "satisfeito.bson" par acc
# Flux.loadparams!(model, par)

verify = gpu.(verify)


model = gpu(model)
data_train = gpu.(data_train)
evalcb = () -> @show(accuracy(data_train, model))
opt = ADAM(3e-4)

lossmean(data_train, model)
acc = accuracy(verify, model)
accuracy(data_train, model)


@epochs 1 Flux.train!(loss, Flux.params(model), data_train, opt, cb = evalcb)

par = params(model)
typeof(acc)
@save "satisfeito_deeper.bson" par acc

#testing

function testify(data) 
    rows = [[i for i in data[j]] for j in 1:length(data)]
    stops = Array{Float32}(undef,25, 25,1, length(data))
    for (i,row) in enumerate(rows)
        two_squares = permutedims(reshape(row,25,25,1,1),(2,1,3,4))
        stops[:,:,:,i] = two_squares[:,:,:,1]
    end
    return stops
end
testmode!(model, true)
test = DataLoader(testify(test_data))
test = gpu.(test)

results = cpu.([model(x) for x in test])

final = [results[i] .> 0.5 for i in 1:length(results)]

##Reshaping for submission
lala = reshape.(final,25,25)
lala2 = transpose.(lala)
lala3 = reshape.(lala2,625)
final = cpu(lala3)
final2 = Array{Array{Int32,1}}(final)

a = copy(test_data.names)
pushfirst!(a,:id)

test_ids = (CSV.File("testUndefVarError: Zygote not defined.csv", select=[:id]))
test_ids.id
for (i,row) in enumerate(final2)
    pushfirst!(row,test_ids.id[i])
end

##Saving
final3 = hcat(final2...)'
dfinal = DataFrame(final3,  a)
CSV.write("submission.csv", dfinal)



