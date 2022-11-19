from Drive import *
import os
from sklearn.model_selection import train_test_split



path = "/Users/admin/Desktop/Self-Drive/Recording"



### Step 1

df = read_csv(path)
# df, center, steering = read_csv(path)
# imagepath = np.asarray(center)
# stearing =  np.asarray(steering)

# print(imagepath[0])
# print(sterring[0])



# ### Step 2
drow_histogram(df, display=False)


### Step 3
imagepath , steering = laod_data(path, df)



### Step 4
xtrain,xtest,ytrain,ytest = train_test_split(imagepath, steering, test_size=0.2, random_state=5)
print("xtrain shape: " , xtrain.shape)
print("xtest shape: " , xtest.shape)
print("ytrain shape: " , ytrain.shape)
print("ytest shape: " , ytest.shape)

### Step 8
model = createmodel()
model.summary()

### Step 9
history = model.fit(batchGen(xtrain, ytrain, 10, 1), steps_per_epoch = 20, epochs = 100,
                    validation_data = batchGen(xtest, ytest, 10, 0), validation_steps = 20)

### Step 10

model.save('modell.h5')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0,1])
plt.title('loss')
plt.xlabel("Epoch")
plt.show()