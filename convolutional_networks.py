"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from a2_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, ReLU, adam, gtid


def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modfiy the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) // stride
          W' = 1 + (W + 2 * pad - WW) // stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the convolutional forward pass.                  #
        # Hint: you can use function torch.nn.functional.pad for padding.  #
        # You are NOT allowed to use anything from torch.nn in other places. #
        ####################################################################
        # Replace "pass" statement with your code

        # 1) Get the input shape, filter shape, stride, and pad
        N = x.shape[0]
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]

        F = w.shape[0]
        #C = w.shape[1]
        HH = w.shape[2]
        WW = w.shape[3]

        stride = conv_param["stride"]
        pad = conv_param["pad"]

        # 2) Pad input
        x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad))

        # 3) Calculate H' and W'
        H_ = 1 + (H + 2 * pad - HH) // stride
        W_ = 1 + (W + 2 * pad - WW) // stride


        # 3) Initialize output
        out = torch.zeros((N,F,H_,W_))
        
        # 4) Loop batch_size, loop filters, then x and y
        for n in range(N):
          for f in range(F):
            for k in range(0, H_ * stride, stride):
                for m in range(0, W_ * stride, stride):

                    # 5) Get the receptive field (input image square to convolve on)
                    r_field = x_pad[n, :, k:k+HH, m:m+WW]
                    
                    # Perform convolution
                    out[n, f, k//stride, m//stride] = torch.sum(r_field * w[f]) + b[f]


        #####################################################################
        #                          END OF YOUR CODE                         #
        #####################################################################
        cache = (x, w, b, conv_param)
        return out, cache


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the max-pooling forward pass                     #
        ####################################################################
        # Replace "pass" statement with your code
        
        # 1) Get the input shape, pool shape, stride
        N = x.shape[0]
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]

        p_height = pool_param["pool_height"]
        p_width = pool_param["pool_width"]
        stride = pool_param["stride"]

        # 2) Get H_ and W_
        H_ = 1 + (H - p_height) // stride
        W_ = 1 + (W - p_width) // stride

        # 3) Initialize output
        out = torch.zeros((N,C,H_,W_))

        # 4) Loop batch_size, loop filters, then x and y
        for n in range(N):
          for c in range(C):
            for k in range(0, H_ * stride, stride):
                for m in range(0, W_ * stride, stride):

                    # 5) Get the receptive field (input image square to max pool on)
                    r_field = x[n, c, k:k+p_height, m:m+p_width]
                    
                    # Perform max pooling
                    out[n, c, k//stride, m//stride] = torch.max(r_field)

 
        

        ####################################################################
        #                         END OF YOUR CODE                         #
        ####################################################################
        cache = (x, pool_param)
        return out, cache

class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                         torch.zeros_like(layer.weight), \
                         torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A custom layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        out = None
        cache = None
        ######################################################################
        # TODO: Refer to the Conv_ReLU class above and implement a similar   #
        # forward function that performs first a convolution, then a ReLU    #
        # and lastly a pool                                                  #
        ######################################################################
        # Replace "pass" statement with your code

        # 1) Conv_ReLU forward pass
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        b, relu_cache = ReLU.forward(a)

        # 2) Max Pool forward pass
        out, pool_cache = FastMaxPool.forward(b, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)

        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool
        convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db

class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
  

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ######################################################################
        ############################  TODO  ##################################
        # Initialize weights and biases for the three-layer convolutional network.
        # Weights should be initialized from a Gaussian distribution centered at 0.0
        # with a standard deviation equal to 'weight_scale'. Biases should be initialized to zero.
        # Store all weights and biases in the 'self.params' dictionary.

        # For the convolutional layer, use the keys 'W1' and 'b1'.
        # For the hidden linear layer, use keys 'W2' and 'b2'.
        # For the output linear layer, use keys 'W3' and 'b3'.

        # IMPORTANT: You can assume that the padding and stride of the first convolutional layer
        # are chosen so that the width and height of the input are preserved. Check the
        # 'loss()' function for more details on this.
        # HINT: Use input data (conv_param, pool_param, input_dim, filter_size etc) 
        #       to figure out the dimensions of weights and biases
        ######################################################################
        # Replace "pass" statement with your code
        
        # 1) Get input shape
        C = input_dims[0]
        H = input_dims[1]
        W = input_dims[2]

        # 2) Initialize weights and biases
        
        # 2.1) Conv_ReLu_Pool layer
        self.params['W1'] = torch.normal(0, weight_scale, size =(num_filters, C, filter_size, filter_size)).to(device, dtype)
        self.params['b1'] = torch.zeros(num_filters,device = device,dtype = dtype)
        
        # 2.2) Hidden Layer
        self.params['W2'] = torch.normal(0, weight_scale, size=(num_filters * H * W // 4, hidden_dim)).to(device, dtype)
        self.params['b2'] = torch.zeros(hidden_dim,device = device,dtype = dtype)

        # 2.3) Output layer
        self.params['W3'] = torch.normal(0, weight_scale, size=(hidden_dim, num_classes)).to(device, dtype)
        self.params['b3'] = torch.zeros(num_classes,device = device, dtype = dtype)

        #print("Passed Initialization")
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Inputs:
        - X: Tensor of input data
        - y: int64 Tensor of labels, of shape (N,). y[i] gives the label for X[i].
        """

        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        # pass conv_param to the forward pass for the convolutional layer
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for the three-layer conv net.     #
        # Computing the class scores for X and storing them in the scores    #
        # variable. Store outputs of all the layers since they will be used  #
        # in computing gradients (see last TODO for this class)              #                                                         #
        #                                                                    #
        # HINT: Use forward functions of the custom layers that you have     #
        # already implemented before.                                        #
        ######################################################################
      
        # Replace "pass" statement with your code
        
        # 1) Conv_ReLU_pool forward pass
        X = X.to(self.dtype)
        a, cache1 = Conv_ReLU_Pool.forward(X, self.params['W1'], self.params['b1'], conv_param, pool_param)
        
        #print("Passed Conv_ReLU_Pool")

        # 2) Hidden layer - Linear_ReLU forward pass
        b, relu_cache = Linear_ReLU.forward(a,self.params['W2'],self.params['b2'])
        cache2 = (cache1, relu_cache)
       
        #print("Passed ReLU")
        
        # 3) Output layer - Linear forward pass
        out, fc_cache = Linear.forward(b,self.params['W3'],self.params['b3'])
        cache3 = (cache1, cache2, fc_cache)
      
        #print("Passed Output Layer")
        scores = out
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################

        if y is None:
            return scores

        # Following is the implementation of the backward pass for the 
        # three-layer convolutional network.
        loss, grads = 0.0, {}
        loss, dout = softmax_loss(scores, y)

        ####################################################################
        # TODO: Incorporate L2 regularization into loss
        # 
        ####################################################################
        # Replace "pass" statement with your code
        i = 3
        # 1) Square each weight
        w1_squared = torch.square(self.params['W1'])
        w2_squared = torch.square(self.params['W2'])
        w3_squared = torch.square(self.params['W3'])

        # 2) Sum each squared vector
        w1_sum = torch.sum(w1_squared)
        w2_sum = torch.sum(w2_squared)
        w3_sum = torch.sum(w3_squared)
        w_sum1 = torch.add(w1_sum,w2_sum)
        w_sum2 = torch.add(w_sum1, w3_sum)

        # 3) Sum and multiply by lambda
        loss = loss + self.reg * w_sum2
        
        ###################################################################
        #                             END OF YOUR CODE                    #
        ###################################################################

        ####################################################################
        # TODO: Assign the 2nd output variables (cache) of your custom layers to the given below 
        # variables so that it can be used in calculating gradients below
        ####################################################################
        cache_L = fc_cache # replace None with 2nd output of forward for linear layer
        cache_LR = relu_cache # replace None with 2nd output of forward for linear_relu layer
        cache_CRP = cache1 # replace None with 2nd output of forward for conv_relu_pool layer
        ###################################################################
        #                             END OF YOUR CODE                    #
        ###################################################################

        #Following code calculates gradients
        last_dout, dw, db  = Linear.backward(dout, cache_L)
        grads['W{}'.format(i)] = dw + 2*self.params['W{}'.format(i)]*self.reg
        grads['b{}'.format(i)] = db
        i-=1
        last_dout, dw, db  = Linear_ReLU.backward(last_dout, cache_LR)
        grads['W{}'.format(i)] = dw + 2*self.params['W{}'.format(i)]*self.reg
        grads['b{}'.format(i)] = db
        i-=1
        last_dout, dw, db  = Conv_ReLU_Pool.backward(last_dout, cache_CRP)
        grads['W{}'.format(i)] = dw + 2*self.params['W{}'.format(i)]*self.reg
        grads['b{}'.format(i)] = db

        return loss, grads


def create_convolutional_solver_instance(data_dict, dtype, device):

    #### put your final hyperparameters here ####
    num_filters = 64
    filter_size = 7
    hidden_dim = 200
    reg = 0.01
    weight_scale = 1e-3
    learning_rate = 1e-3
    num_epochs = 30
    batch_size = 128
    update_rule = adam
    #############################################

    input_dims = data_dict['X_train'].shape[1:]
    model = ThreeLayerConvNet(num_filters=num_filters, filter_size=filter_size, 
                              hidden_dim=hidden_dim, reg=reg, weight_scale=weight_scale, 
                              dtype=torch.float, device='cpu')

    solver = Solver(model, data_dict,
                    num_epochs=num_epochs, batch_size=batch_size,
                    update_rule=adam,
                    optim_config={
                      'learning_rate': learning_rate,
                    },
                    device='cpu')
    return solver
