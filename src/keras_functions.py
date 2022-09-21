from keras.models import Sequential #Para poder definir un modelo secuencial
from keras.layers import Dense, BatchNormalization #Para poder usar capas densas
from keras.optimizers import SGD, Adam
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
import helper

def show_learning(x_train, y_train, x_test, y_test, optimizer='Adam', batch_size=32,epochs=10,filename='movie.mp4'):
    model=Sequential() #Queda definido el modelo sequencial
    #model.add(BatchNormalization())
    model.add(Dense(1, input_shape=(2,),activation='sigmoid',use_bias=False,kernel_initializer='zeros',bias_initializer='zeros'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy'])
    learning_curve=helper.plot_learning_curve(plot_interval=1, 
                                              evaluate_interval=None, 
                                              x_val=None, 
                                              y_val_categorical=None,
                                              epochs=epochs)
    def get_weights(model):
        w1=model.layers[0].get_weights()[0][0][0]
        w2=model.layers[0].get_weights()[0][1][0]
        return [w1,w2]
    def set_weights(model, w1, w2):
        weights=model.layers[0].get_weights()
        weights[0][0][0]=w1
        weights[0][1][0]=w2
        model.layers[0].set_weights(weights)
    log_weights=helper.log_weights(get_weights)
    set_weights(model=model,w1=0.1,w2=-0.14)
    X_loss=x_train[0:100,:]
    y_loss=y_train[0:100]
    
    history=model.fit(x_train, y_train, validation_data=[x_test,y_test],
                  epochs=epochs,verbose=0, batch_size=batch_size,shuffle = True,
                  callbacks=[learning_curve, log_weights])
    loss=list()
    
    w1_mesh,w2_mesh,J=helper.plot_loss_surface(X_loss, y_loss, model, set_weights, [-1,0.5], [-0.2,0.5],20,plot=True)
    for w in log_weights.weights:
        loss.append(helper.get_loss(w[0],w[1],model,X=X_loss,y=y_loss,set_weights=set_weights))
    data = np.hstack([np.array(log_weights.weights),np.array(loss).T.reshape(-1,1)]).T
    fig, ax = plt.subplots()
    CS = ax.contourf(w1_mesh, w2_mesh, J, 100, cmap=plt.cm.coolwarm)
    line, = ax.plot([], [],'k')
    def animate(i):
        line.set_data(data[0,:i],data[1,:i])  # update the data
        if i % int(len(loss)/100)==0:
            porc=int(i/len(loss)*100)
            print("\r {}%".format(porc),end="")
        return line,
    ani = animation.FuncAnimation(fig, animate, len(loss),
                              interval=25, blit=False)
    ani.save(filename)
    plt.show()
    