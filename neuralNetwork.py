import numpy as np
import pickle
import copy
#-------------------------------- dense layer------------------------------------------------------------------------------------------------
#Dense layer

class Layer_Dense():

    #layer initialization
    def __init__(self, n_inputs, n_neurons, weight_regularizer_L1 =0,
                 weight_regularizer_L2 =0 , bias_regularizer_L1 =0,bias_regularizer_L2=0 ):
        
        

        #Initialize weights and  biases
        self.weights = 0.01 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        #set regularization strength
        self.weight_regularizer_L1 = weight_regularizer_L1
        self.weight_regularizer_L2 = weight_regularizer_L2
        self.bias_regularizer_L1 = bias_regularizer_L1
        self.bias_regularizer_L2 = bias_regularizer_L2


        #forward pass

    def forward(self, inputs,training):
        self.inputs = inputs
        #calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dvalues):
        #gardients on parameters
        self.dweights= np.dot(self.inputs.T, dvalues)
        self.dbiases= np.sum(dvalues, axis=0, keepdims=True)
        #gardients on regularization
        if self.weight_regularizer_L1> 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights <0 ]= -1
            self.dweights += self.weight_regularizer_L1 * dL1
        if self.weight_regularizer_L2> 0:
            self.dweights += 2* self.weight_regularizer_L2 * self.weights
        if self.bias_regularizer_L1> 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases <0 ]= -1
            self.dbiases += self.bias_regularizer_L1 * dL1
        if self.bias_regularizer_L2> 0:
            self.dbiases += 2* self.bias_regularizer_L2 * self.biases
        #gardient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_parameters(self):
        return self.weights,self.biases
    def set_parameters(self,weights,biases):
        self.weights = weights
        self.biases=biases
        



     




#--------------------------------Dropout regularization------------------------------------------------------------------------------------------------

class Layer_Dropout:
    #init
    def __init__(self,rate):
        self.rate= 1-rate

    #forward pass
    def forward(self,inputs,training):
        #save input values
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return
        
        #generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate,size=inputs.shape) / self.rate

        #apply mask to output values
        self.output = inputs * self.binary_mask

    #backward pass
    def backward(self,dvalues):
        #gardient on values
        self.dinputs = dvalues* self.binary_mask



#-------------------------------- Layer input------------------------------------------------------------------------------------------------

class Layer_Input:
    def forward(self, inputs,training):
        self.output = inputs           
    
        

        
#--------------------------------relu activation------------------------------------------------------------------------------------------------
#RELU activation
            
class Activation_Relu:
    #forward pass
    def forward(self,inputs,training):
        #remember the input values
        self.inputs= inputs
        #calculate output values from inputs
        self.output= np.maximum(0,inputs)
    def backward(self,dvalues):
        #since we need to modify the original variables
        #let's make a copy of the values first
        self.dinputs= dvalues.copy()

        #zero gardient where input values were negative
        self.dinputs[self.inputs <=0] =0

    def predictions(self,outputs):
        return outputs
        
        
#--------------------------------softmax activation------------------------------------------------------------------------------------------------
#Softmax activation

class Activation_Softmax:
    #forward pass
    def forward(self,inputs,training):
        self.inputs = inputs
        
        exp_values =  np.exp(inputs- np.max(inputs, axis=1, keepdims =True) )
       
        
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        
        self.output = probabilities
    def backward(self, dvalues):

        #create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        #enumerate outputs and gardietns
        for index , (single_output, single_dvalues) in enumerate(zip(self.output,dvalues)):
            #flatten output array
            single_output = single_output.reshape(-1,1)
            #calculate Jaconian matric of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            #calculate sample-wise gardient
            #and add it to the array of sample gardient
            self.dinputs[index] = np.dot(jacobian_matrix , single_dvalues)

    def predictions(self,outputs):
        return np.argmax(outputs,axis=1)

#--------------------------------sigmoid activation------------------------------------------------------------------------------------------------

class Activagion_Sigmoid:
    #forward pass
    def forward(self,inputs,training):
        self.inputs = inputs
        self.output = 1/ (1+np.exp(-inputs))
    #backward pass
    def backward(self,dvalues):
        #derivative of sigmoid
        self.dinputs = dvalues * (1- self.output) * self.output
    def predictions(self, outputs):
        return (outputs > 0.5) *1

    
#--------------------------------Linear activation------------------------------------------------------------------------------------------------
class Activation_Linear:
    #forward pass
    def forward(self,inputs,training):
        self.inputs = inputs
        self.output =inputs
    def backward(self, dvalues):
        #derivative is equal to 1
        self.dinputs = dvalues.copy()
    def predictions(self,outputs):
        return outputs
#--------------------------------optimizer SGD------------------------------------------------------------------------------------------------
class optimizer_SGD:
    #initialize optimizer
    def __init__(self,learning_rate=1.0, decay=0, momentum =0. ):
        self.learning_rate= learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations =0
        self.momentum= momentum
    #call once before parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * ( 1.0 / (1.0 + self.decay * self.iterations)) 

    #update parameters
    def update_params(self,layer):
        #if we use momentum
        if self.momentum:

            #if layer doesn't contain momentum arrays, create them filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                #if there is no momentum array for weights
                #the array doesn't exist for biases also
                layer.bias_momentums = np.zeros_like(layer.biases)

            #build weights updates with momentum
            #and update multiplied by retain factor and with current gardients
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            #bias update
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        #vanilla sgd updtes (as before momentum)
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = - self.current_learning_rate* self.dbiases




        #update weights and biases using either vanilla or momentum    
        layer.weights += weight_updates
        layer.biases += bias_updates

    #call once after any update
    def post_update_params(self):
        self.iterations+=1
        
        
        


#--------------------------------optimizer Adagrad------------------------------------------------------------------------------------------------



class optimizer_Adagrad:
    #initialize optimizer
    def __init__(self,learning_rate=1.0, decay=0.0, epsilon =1e-7 ):
        self.learning_rate= learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations =0
        self.epsilon =epsilon
    #call once before parameter updates
    def pre_update_params(self):
        if self.decay:
            temp3= ( 1.0 / (1.0 + self.decay * self.iterations)) 
            self.current_learning_rate = self.learning_rate * temp3

    #update parameters
    def update_params(self,layer):
       

        #if layer doesn't contain momentum arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache= np.zeros_like(layer.biases)
        #update cache with squared current gardients 
        layer.weight_cache+= layer.dweights **2
        layer.bias_cache+= layer.dbiases **2
        #vanilla sgd parameter updates + normalization with squre rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)


    #call once after any update
    def post_update_params(self):
        self.iterations+=1 
#--------------------------------optimizer RMS------------------------------------------------------------------------------------------------
class optimizer_RMSprop:
    #initialize optimizer
    def __init__(self,learning_rate=0.001, decay=0.0, epsilon =1e-7 , rho= 0.9):
        self.learning_rate= learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations =0
        self.epsilon =epsilon
        self.rho = rho
    #call once before parameter updates
    def pre_update_params(self):
        if self.decay:
            temp3= ( 1.0 / (1.0 + self.decay * self.iterations)) 
            self.current_learning_rate = self.learning_rate * ( 1.0 / (1.0 + self.decay * self.iterations)) 

    #update parameters
    def update_params(self,layer):
       

        #if layer doesn't contain momentum arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache= np.zeros_like(layer.biases)
        #update cache with squared current gardients 
        layer.weight_cache = self.rho * layer.weight_cache + (1- self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1- self.rho) * layer.dbiases**2
        #vanilla sgd parameter updates + normalization with squre rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)


    #call once after any update
    def post_update_params(self):
        self.iterations+=1


#--------------------------------optimizer Adam------------------------------------------------------------------------------------------------
class optimizer_Adam:
    #initialize optimizer
    def __init__(self,learning_rate=0.001, decay=0.0, epsilon =1e-7 , beta_1= 0.9, beta_2= 0.999):
        self.learning_rate= learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations =0
        self.epsilon =epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2        
    #call once before parameter updates
    def pre_update_params(self):
        if self.decay:
            temp3= ( 1.0 / (1.0 + self.decay * self.iterations)) 
            self.current_learning_rate = self.learning_rate * ( 1.0 / (1.0 + self.decay * self.iterations)) 

    #update parameters
    def update_params(self,layer):
       

        #if layer doesn't contain momentum arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums= np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums= np.zeros_like(layer.biases)
            layer.bias_cache= np.zeros_like(layer.biases)
        #get corrected momentum
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1- self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1- self.beta_1) * layer.dbiases
        weight_momentums_corrected = layer.weight_momentums/ (1-self.beta_1 ** (self.iterations+1))
        bias_momentums_corrected = layer.bias_momentums/ (1-self.beta_1 ** (self.iterations+1))


        #update cache with squred current gardients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1- self.beta_2) * layer.dweights **2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1- self.beta_2) * layer.dbiases **2

        #get correct cache
        weight_cache_corrected = layer.weight_cache / (1- self.beta_2 ** (self.iterations +1 ))
        bias_cache_corrected = layer.bias_cache / (1- self.beta_2 ** (self.iterations +1 ))        
        #vanilla sgd parameter updates + normalization with squre rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(bias_cache_corrected) + self.epsilon)


    #call once after any update
    def post_update_params(self):
        self.iterations+=1
        
#--------------------------------loss calss------------------------------------------------------------------------------------------------
            
#common loss class
class Loss:
    #calculate the data and regularization losses
    #given model output and ground truth values

    def regularization_loss(self):
        # 0 by default
        regularization_loss =0
        #L1 regularization- weights
        #calculate only when factor greater than 0
        for layer in self.trainable_layers:
            
            if layer.weight_regularizer_L1 > 0 :
                regularization_loss += layer.weight_regularizer_L1 * np.sum(np.abs(layer.weights))
            #L2 
            if layer.weight_regularizer_L2 >0 :
                regularization_loss += layer.weight_regularizer_L1 * np.sum(np.abs(layer.weights * layer.weights))
            
        
        #L1 regularization bias 
        #calculate only when factor greater than 0
            if layer.bias_regularizer_L1 > 0 :
                regularization_loss += layer.bias_regularizer_L1 * np.sum(np.abs(layer.biases))
            if layer.bias_regularizer_L2 >0 :
                regularization_loss += layer.bias_regularizer_L1 * np.sum(np.abs(layer.biases * layer.biases))
            return regularization_loss
    def remember_trainable_layers(self,trainable_layers):
        self.trainable_layers = trainable_layers        
    def calculate(self,output,y,*,include_regularization=False):

        #calculate sample loss
        sample_losses = self.forward(output, y)

        #calculate mean loss
        data_loss = np.mean(sample_losses)

        #add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)       

        if not include_regularization:
            return data_loss

        #return loss
        return data_loss, self.regularization_loss()

    #calc accumulated loss
    def calculate_accumulated(self,*,include_regularization =False):
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()
    #reset variables for accumulated pass
    def new_pass(self):
        self.accumulated_sum =0
        self.accumulated_count=0
        
    
            



#--------------------------------loss categorical cross entropy------------------------------------------------------------------------------------------------

#Cross_entropy loss
class Loss_categoricalCrossEntropy(Loss):
    #forward pass
    def forward(self,y_pred,y_true):
        #Number of samples in a batch
        samples = len(y_pred)

        #clip data to prevent division by 0
        #clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred , 1e-7, 1- 1e-7)


        #probabilities for target values only if categorical labels
        if(len(y_true.shape) ==1 ):
            correct_confidences = y_pred_clipped[range(samples), y_true]
        #mask values - only for one-hot encoded labels
        elif (len(y_true.shape) == 2 ):
            correct_confidences = np.sum( y_pred_clipped * y_true , axis =1 )
            #losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self,dvalues, y_true):
        #number of samples
        samples = len(dvalues)
        #number of labels in every sample
        #we'll use the first sample to count them
        labels = len(dvalues[0])

        #if labels are sparse, turn then into one-hot vector
        if len(y_true.shape) ==1:
            y_true= np.eye(labels)[y_true]
        #calculate gardient
        self.dinputs = - y_true / dvalues

        #normalize gardient
        self.dinputs = self.dinputs / samples

#--------------------------------activation soft-max with loss------------------------------------------------------------------------------------------------
class Activation_softmax_Loss_CategoricalCrossentropy():
    #create activation and loss function objects
    '''
    def __init__(self):
        self.activation= Activation_Softmax()
        self.loss= Loss_categoricalCrossEntropy()
    #forward pass
    def forward(self, inputs, y_true):
        #output layer's activation function
        self.activation.forward(inputs)

        #set the output
        self.output = self.activation.output
        #calculate and return loss value
        return self.loss.calculate(self.output, y_true)
        '''
    #backward pass
    def backward(self, dvalues , y_true):
        #number of samples
        samples = len(dvalues)
        #if labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape)==2:
            y_true = np.argmax(y_true,axis=1)
        #copy so we can safely modify
        self.dinputs = dvalues.copy()
        #calculate gardient
        self.dinputs[range(samples), y_true] -=1

        #normalize gardient
        self.dinputs = self.dinputs / samples


#--------------------------------Loss Binary Cross Entropy------------------------------------------------------------------------------------------------

class Loss_BinaryCrossentropy(Loss):
    def forward(self,y_pred,y_true):
        #clip data to prevent devide by 0
        y_pred_clipped = np.clip(y_pred , 1e-7,1-1e-7)
        sample_losses = - (y_true * np.log(y_pred_clipped) + (1- y_true) * np.log(1- y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=1)
        return sample_losses
    def backward(self,dvalues, y_true):
        sample = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, 1e-7 , 1-1e-7)
        #calculate gardient
        self.dinputs = -(y_true / clipped_dvalues - (1- y_true) / (1-clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples




#--------------------------------loss mean squred calss------------------------------------------------------------------------------------------------
class Loss_MeanSquaredError(Loss):
    def forward(self,y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        #return losses
        return sample_losses
    def backward(self, dvalues, y_true):
        #number of samples
        samples= len(dvalues)
        #number of outputs in every sample
        outputs = len(dvalues[0])

        #gardient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs

        #normalize gardient
        self.dinputs = self.dinputs/ samples
#--------------------------------loss mean absolute calss------------------------------------------------------------------------------------------------
class Loss_MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true- y_pred), axis = -1 )
        return sample_losses
    def backward(self, dvalues , y_true):
        samples = len(dvalues)
        outputs= len(dvalues[0])
        self.dinputs= np.sign(y_true- dvalues) / outputs
        self.dinputs = self.dinputs / samples
            





#--------------------------------calc accuracy------------------------------------------------------------------------------------------------
class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions,y)
        accuracy = np.mean(comparisons)
        self.accumulated_sum+= np.sum(comparisons)
        self.accumulated_count+= len(comparisons)
        return accuracy
    #calc accumulated loss
    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy
    #reset variables for accumulated pass
    def new_pass(self):
        self.accumulated_sum =0
        self.accumulated_count=0

#--------------------------------accuracy categorical------------------------------------------------------------------------------------------------

class Accuracy_Categorical(Accuracy):
    def init(self,y):
        pass
    def compare(self,predictions,y):
        if len(y.shape) ==2:
            y= np.argmax(y, axis=1)
        return predictions ==y
#--------------------------------accuracy regression------------------------------------------------------------------------------------------------    
class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None
    def init(self,y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y)/ 250
    def compare(self, predictions, y ):
        return np.absolute(predictions -y) < self.precision
#-------------------------------- Model class------------------------------------------------------------------------------------------------
class Model:
    def __init__(self):
        #create a list of network objects
        self.layers = []
        self.softmax_classifier_output = None

    #add objects to the model
    def add(self,layer):
        self.layers.append(layer)

    #set loss and optimizer
    def set(self,*, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer= optimizer
        if accuracy is not None:
            self.accuracy= accuracy
    #finalize the model
    
    def finalize(self):
        #create and set the input layer 
        self.input_layer = Layer_Input()

        #count all the objecs
        layer_count =len(self.layers)
        self.trainable_layers= []
        
        for i in range(layer_count):


            if i ==0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i<layer_count -1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
               
            if hasattr(self.layers[i],'weights'):
                self.trainable_layers.append(self.layers[i])
            self.loss.remember_trainable_layers(self.trainable_layers)
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_categoricalCrossEntropy):
            self.softmax_classifier_output = Activation_softmax_Loss_CategoricalCrossentropy()    
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)
                
            
    def train(self, X,y,*, epochs=1, batch_size = None ,print_every=1, validation_data=None):
        #initialize accuracy object
        self.accuracy.init(y)
        #default value if batch size is not set
        train_steps =1
        #if there is validation data passed set default number of steps for val as well
        
        if validation_data is not None:
            validation_steps=1
            #for better readability
            X_val,y_val = validation_data
        #calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            #because divideing round down we'll add 1 to get any extra data 
            if train_steps * batch_size < len(X):
                train_steps +=1
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps +=1

        for epoch in range(1,epochs+1):
            print(f'epoch: {epoch}')
            self.loss.new_pass()
            self.accuracy.new_pass()
            for step in range(train_steps):
                #if batch size is not set
                if batch_size is None:
                    batch_X=X
                    batch_y=y
                #else slice the batch
                else:
                    batch_X=X[step*batch_size :(step+1)*batch_size]
                    batch_y=y[step*batch_size :(step+1)*batch_size]
                
            output =self.forward(batch_X,training=True)
            data_loss , regularization_loss = self.loss.calculate(output,batch_y, include_regularization= True)
            loss = data_loss + regularization_loss
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, batch_y)
            self.backward(output,batch_y)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()
            if not step % print_every or step==train_steps-1:
                print(f'step: {step} , ' +
                      f'acc: {accuracy: .3f}, ' +
                      f'loss: {loss: .3f}, '+
                      f'data_loss: {data_loss: .3f}' +
                      f'reg_loss: {regularization_loss:.3f} , '+
                      f'lr: {self.optimizer.current_learning_rate}')
        #get and print epoch loss and acc
                
        epoch_data_loss , epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
        epoch_loss= epoch_data_loss + epoch_regularization_loss
        epoch_accuracy= self.accuracy.calculate_accumulated()
        print(f'training , ' +
                      f'acc: {epoch_accuracy: .3f}, ' +
                      f'loss: {epoch_loss: .3f}, '+
                      f'data_loss: {epoch_data_loss: .3f}' +
                      f'reg_loss: {epoch_regularization_loss:.3f} , '+
                      f'lr: {self.optimizer.current_learning_rate}')
        if validation_data is not None:
            self.evaluate(*validation_data,batch_size=batch_size)
        
        
    #evaluate the model using passed in dataset
    def evaluate(self,X_val,y_val,*,batch_size=None):
        validation_steps=1
        if batch_size is not None:
            validation_steps = len(X_val)// batch_size
            if validation_steps *batch_size < len(X_val):
                validation_steps+=1
        self.loss.new_pass()
        self.accuracy.new_pass()
        for step in range(validation_steps):
            if batch_size is None:
                batch_X= X_val
                batch_y= y_val
            else:
                batch_X=X_val[step*batch_size :(step+1)*batch_size]
                batch_y=y_val[step*batch_size :(step+1)*batch_size]
            
            output= self.forward(batch_X,training=False)
            self.loss.calculate(output,batch_y)
            predictions= self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions,batch_y)
        validation_loss= self.loss.calculate_accumulated()
        validation_accuracy= self.accuracy.calculate_accumulated()        
        print(f'validation, '+
                f'acc: {validation_accuracy:.3f}, '+
                f'loss: {validation_loss: .3f}, '
            )
        
        
        
           
    def forward(self, X,training):
        self.input_layer.forward(X,training)
        for layer in self.layers:
            layer.forward(layer.prev.output,training)
        return layer.output
    def backward(self,output,y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output,y)
            self.layers[-1].dinputs= self.softmax_classifier_output.dinputs
        
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
            self.loss.backward(output,y)
            
            for layer in reversed(self.layers):
                layer.backwards(layer.next.dinputs)
    def get_parameters(self):
        parameters=[]
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters
    def set_parameters(self,parameters):
        for parameter_set, layer in zip(parameters,self.trainable_layers):
            layer.set_parameters(*parameter_set)
    def save_parameters(self,path):
        with open(path,'wb') as f:
            pickle.dump(self.get_parameters(),f)
    def load_parameters(self,path):
        with open(path,'rb') as f:
            self.set_parameters(pickle.load(f))
    def save(self,path):
        model = copy.deepcopy(self)
        model.loss.new_pass()
        model.accuracy.new_pass()
        model.input_layer.__dict__.pop('output',None)
        model.loss.__dict__.pop('dinputs',None)
        for layer in model.layers:
            for property in ['inputs','outputs','dinputs','dweights','dbiases']:
                layer.__dict__.pop(property,None)
        with open(path,'wb') as f:
            pickle.dump(model,f)
    @staticmethod
    def load(path):
        with open(path,'rb') as f:
            model = pickle.load(f)
        return model
    def predict(self,X,*,batch_size=None):
        prediction_steps=1
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps+=1
        output=[]
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X=X
            else:
                batch_X= X[step * batch_size:(step+1)*batch_size]
            batch_output = self.forward(batch_X,training=False)
            output.append(batch_output)
        return np.vstack(output)
    '''
#---------------------------------------------------------------------------------------------------------------
import pandas as pd
import spacy
df = pd.read_csv("CQFinal.csv")
category = df.category
category_to_number = {'Login Issues':0
                      ,'forgot password' : 1
                      ,'registration issues':2
                      ,'payment issues':3
                      ,'final grades':4
                      ,'Transcript':5
                      ,'submitting error':6
                      ,'Online Learning':7
                      ,'Library issues':8
                      ,'contact IT':9
                      ,'wifi issues':10
                      ,'Mobile app':11
                      ,'apply for graduation':12
                      ,'final grades':13
                      ,'drop course':14
                      ,'class info ':15
                      ,'Registeration Issue':16
                      ,'Vpn':17
                      ,'guests policy':18
                      ,'change contact info ':19
                      ,'E-mail Issue':20
                      ,'Security and Privacy':21}
df['labeled_category'] = list(map(lambda x: category_to_number[x], category))

nlp= spacy.load("en_core_web_lg")
def process_text(text):
    # Parse the text with SpaCy
    doc = nlp(text)
    # Remove stop words and convert to lowercase
    processed_text = [token.text.lower() for token in doc if not token.is_stop]
    # Join the processed tokens back into a string
    return ' '.join(processed_text)

# Apply the process_text function to the 'text' column and save the changes back
#df['Questions_changed'] = df['Question'].apply(process_text)


df['Vector'] = df['Question'].apply(lambda Question : nlp(Question).vector)




from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(
    df['Vector'].values,
    df['labeled_category'],
    test_size=0.2,
    random_state = 2022,
    stratify = df['labeled_category'] 
)
import numpy as np

# Assuming X_train and X_test are your numpy arrays
# Flatten all elements in X_train
X_train = np.array([np.array(x).flatten() for x in X_train])

# Flatten all elements in X_test
X_test = np.array([np.array(x).flatten() for x in X_test])





X_train= (X_train.reshape(X_train.shape[0],-1).astype(np.float32))
X_test =  (X_test.reshape(X_test.shape[0],-1).astype(np.float32))
model=Model()

model.add(Layer_Dense(X_train.shape[1],256))
model.add(Activation_Relu())
model.add(Layer_Dense(256,256))
model.add(Activation_Relu())
model.add(Layer_Dense(256,23))
model.add(Activation_Softmax())
model.set(
    loss=Loss_categoricalCrossEntropy(),
    optimizer= optimizer_Adam(decay=1e-4),
    accuracy = Accuracy_Categorical())
model.finalize()
model.train(X_train,y_train,validation_data=(X_test,y_test),epochs=10000,batch_size=512,print_every=100)
parameters = model.get_parameters()
model.save('saved_model/s15')
'''
#-----------------------------------


'''
model=Model()
model.add(Layer_Dense(X_train.shape[1],64))
model.add(Activation_Relu())
model.add(Layer_Dense(64,64))
model.add(Activation_Relu())
model.add(Layer_Dense(64,23))
model.add(Activation_Softmax())
model.set(
    loss=Loss_categoricalCrossEntropy(),
    accuracy = Accuracy_Categorical() )


model.finalize()
model.evaluate(X_test,y_test)
model.load_parameters('saved_model/s4')
model.evaluate(X_test,y_test)
'''
'''

model = Model()
df['processes_input'] = input("how can I help you with")
processed_input1 = df['processes_input'].apply(process_text)
processed_input2 = processed_input1[0]
vector_question = nlp(processed_input2).vector
model= Model.load('saved_model/s14')
pre =model.predict(vector_question)
pre2 = model.output_layer_activation.predictions(pre)
print(pre2)
category = next(key for key, value in category_to_number.items() if value == pre2[0])
print(category)
'''
