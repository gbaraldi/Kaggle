using CSV, CUDA, Flux
using DataFrames
using Flux: onehotbatch,  onecold, logitcrossentropy, throttle, @epochs
using Statistics
using Flux.Data: DataLoader
pwd()
cd("Titanic")



train_data = CSV.File("lalal.csv"; header =1)
test_data = CSV.File("test.csv")
df = DataFrame(train_data)
ytrain= onehotbatch(train_data.Survived, unique(train_data.Survived))

average = mean(skipmissing(train_data.Age))

##
onehot_emb = onehotbatch(train_data.Embarked, unique(train_data.Embarked))
onehot_pclass = onehotbatch(train_data.Pclass, unique(train_data.Pclass))
onehot_sex = onehotbatch(train_data.Sex, unique(train_data.Sex))
# cabin = ismissing.(df_train.Cabin)'
age = train_data.Age
sib = train_data.SibSp'
parch = train_data.Parch'
##
age = map(age)do x
    if ismissing(x)
        x = average
    else
        x = x
    end
end
xtrain = Float16.(vcat(onehot_emb,onehot_pclass,onehot_sex, age', sib, parch))


onehot_emb_t = onehotbatch(test_data.Embarked, unique(train_data.Embarked))
onehot_pclass_t = onehotbatch(test_data.Pclass, unique(test_data.Pclass))
onehot_sex_t = onehotbatch(test_data.Sex, unique(test_data.Sex))
# cabin_t = ismissing.(test_data.Cabin)'
age_t = test_data.Age
sib_t = test_data.SibSp'
parch_t = test_data.Parch'
 
age_t = map(age_t)do x
    if ismissing(x)
        x = average
    else
        x = x
    end
end
xtest = Float16.(vcat(onehot_emb_t,onehot_pclass_t,onehot_sex_t, age_t', sib_t, parch_t))


data_train = DataLoader(xtrain, ytrain, batchsize=50, shuffle=true)

function loss_all(dataloader, model)
    l = 0f0
    for (x,y) in dataloader
        l += logitcrossentropy(model(x), y)
    end
    l/length(dataloader)
end

function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
    end
    acc/length(data_loader)
end

m = Chain(
    Dense(12,32,leakyrelu),
    Dense(32,64,leakyrelu),
    Dense(64,64,leakyrelu),
    Dense(64,32,leakyrelu),
    Dense(32,2,leakyrelu),
    softmax
)

data_train = gpu.(data_train)
# data_test = gpu.(data_train)
m = gpu(m)
loss(x,y) = logitcrossentropy(m(x), y)

## Training
evalcb = () -> @show(loss_all(data_train, m))
opt = ADAM(3e-4)
    
@epochs 10 Flux.train!(loss, params(m), data_train, opt, cb = evalcb)

@show accuracy(data_train, m)

# xtest = gpu.(xtest)

data_test = gpu.(DataLoader(xtest))

values = [m(x) for x in data_test]

final = [i[1] > 0.5 ? 0 : 1 for i in values]

CSV.write("out.csv",(PassengerId = test_data.PassengerId, Survived = final) )
