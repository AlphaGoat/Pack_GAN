import tensorflow as tf
import functools


def doublewrap(function):
    """
    A decorator decorator, allowing the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.

    source: https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator

@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.

    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped function.

    source: https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))

        return getattr(self, attribute)
    return decorator

class Conv2D(object):

    def __init__(self,
                 kernel_shape,
                 num_kernels,
                 in_channels,
                 strides,
                 activation_fn=tf.nn.relu,
                 kernel_initializer=tf.initializers.GlorotUniform,
                 bias_initializer=tf.initializers.GlorotUniform
                 ):

        # If the provided kernel is only n,
        # expand to make kernel shape n x n
        if len(kernel_shape) == 1:
            kernel_shape = (kernel_shape, kernel_shape)

        self.kernel_shape = kernel_shape

        # Initialize kernels (filters)
        self.kernel = WeightVariable((kernel_shape[0],
                                      kernel_shape[1],
                                      in_channels,
                                      num_kernels),
                                      initializer=kernel_initializer,
                                     )

        # Initializee bias
        self.bias = BiasVariable((num_kernels),
                                 initializer=bias_initializer
                                 )


        # If the provided stride is only n,
        # expand to make stride n x n
        if len(strides) == 1:
            strides = (strides, strides)

        self.stride = strides

        self.activation_fn = activation_fn

    #@tf.function
    def call(self, x):

        # Perform convolution operation
        # TODO: figure out how to do this yourself in C++... might be an
        #       interesting side project
        feature_map = tf.nn.conv2d(x,
                                   self.kernel,
                                   strides=self.strides,
                                   padding='SAME'
                                   )
        # Add bias
        feature_map_biased = tf.bias_add(feature_map, self.bias)

        # Return activated feature map
        return self.activation_fn(feature_map_biased)

class BatchNormalization(object):
    """
    Base class for batch normalization layer

    Essentially, just acts as a container for the params
    initializer in the mean, variance, offset, and scaling
    tensors as well as a wrapper for the tf.nn.batch_normalization
    operation. Nothing too fancy here
    """
    def __init__(self,
                 name,
                 summary_update_freq=10):

        # Variables to keep track of name scope of operation
        #self.model_scope = model_scope
        self.batch_norm_name = name

        # Internal variable to keep track of the update frequency for
        # tensorboard summaries
        self.summary_update_freq = summary_update_freq

    def initialize_weights(self, param_shape):
        """
        Initializes tensors to hold mean, variance,
        scale, and bias hyperparameters
        """
        # Tensor for offset parameters
        self.t_beta = WeightVariable(
                            shape=(param_shape,),
                            name='offset_tensor',
                            initializer=tf.zeros,
                            summary_update_freq=self.summary_update_freq,
                            )

        # Tensor for scaling parameters
        self.t_gamma = WeightVariable(
                            shape=(param_shape,),
                            name='scale_tensor',
                            initializer=tf.zeros,
                            summary_update_freq=self.summary_update_freq,
                            )

    def __call__(self, x, step=0):

        # If first step in training loop, initialize weights
        if step == 0:
            input_shape = tf.shape(x)
            param_shape = input_shape[-1]

            self.initialize_weights(param_shape)

        # Calculate the mean and variance over batch dimension of input
        mean, variance = tf.nn.moments(x, axes=[0])

        return tf.nn.batch_normalization(x,
                                         mean=mean,
                                         variance=variance,
                                         offset=self.t_beta(step),
                                         scale=self.t_gamma(step),
                                         variance_epsilon=0.001
                                         )

class WeightVariable(tf.Module):
    """
    Base class for weight parameters to be used in learning layers.
    Allows us to plot statistics for our weight parameters for every
    step of the training process, if desired
    """
    def __init__(self,
                 shape,
                 name,
                 #model_scope,
                 layer_scope,
                 initializer=tf.initializers.zeros,
                 summary_update_freq=None,
                 ):
        super(WeightVariable, self).__init__(name=name)

        # Shape of initializer parameter tensor
        self.shape = shape

        # Variables to keep track of variable name space
        self.name = name

        # variable to keep track of the scope the variable falls under
        self.layer_scope = layer_scope

        # our intitializer
        self.initializer = initializer

        # Update frequency for tensorboard summaries. If 'None',
        # then we are not maintaining a tensorboard for this operation
        assert type(summary_update_freq) == int or type(summary_update_freq) == type(None)
        self.summary_update_freq = summary_update_freq

        self._initialized = False

    def __call__(self, step):

        if not self._initialized:
            # Initialize Weight Variable
            self.initial = tf.Variable(
                self.initializer(self.shape),
                trainable=True,
                name=self.name,
                dtype=tf.float32,
                )
            self._initialized = True
            print("initial.name: ", self.initial.name)
            assert self.initial.name == "%s%s:0" % (self.layer_scope, self.name)

        # create summaries for updated weights
        if self.summary_update_freq:
            if step % self.summary_update_freq == 0:
                variable_summaries(self.initial, step)

        return self.initial

class BiasVariable(tf.Module):
    """
    Base class for bias parameters to be used in learning layers
    """
    def __init__(self,
                 shape,
                 name,
                 layer_scope,
                 initializer=tf.initializers.zeros,
                 summary_update_freq=None,
                 ):
        super(BiasVariable, self).__init__(name=name)

        # Variable keeping track of shape of initializer parameter tensor
        self.shape = shape

        # Variables keeping track of variable name space
        self.name = name

        # variable to keep track of the scope the variable falls under
        self.layer_scope = layer_scope

        # If no initializer is provided, initialize params as zeros
        self.initializer = initializer

        # Frequency with which to update tensorboard summaries
        assert type(summary_update_freq) == int or type(summary_update_freq) == type(None)
        self.summary_update_freq = summary_update_freq

        self._initialized = False

    def __call__(self, step):

        if not self._initialized:
            # Initialize bias parameters
            self.initial = tf.Variable(
                self.initializer(self.shape),
                trainable=True,
                name=self.name,
                dtype=tf.float32,
                )

            self._initialized = True

            assert self.initial.name == "%s%s:0" % (self.layer_scope, self.name)

        # Call variable_summaries for updated weights
        if self.summary_update_freq:
            if step % self.summary_update_freq == 0:
                variable_summaries(self.initial, step)

        return self.initial

class ResidualLayer(object):
    """
    Base class for residual layers as seen in ResNet
    """
    def __init__(self,
                 name,
                 model_scope,
                 filter1_shape,
                 filter2_shape,
                 bias1_shape,
                 bias2_shape,
                 strides_1,
                 strides_2,
                 weight_initializer,
                 num_output_channels):

        self.name = name
        self.model_scope = model_scope

        # Initialize weight variables
        self.filter1_shape = filter1_shape
        self.filter1 = WeightVariable()
        pass

    # TODO: Finish residual layer class
    #       probably will never use it, but it would be
    #       good to have it just in case I want to make
    #       a test model quickly

def variable_summaries(var, step):
    """
    Method for saving summary statistics to TensorBoard
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean, step=step)

        with tf.name_scope('std_dev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar('stddev', stddev, step=step)
        tf.summary.scalar('max', tf.reduce_max(var), step=step)
        tf.summary.scalar('min', tf.reduce_min(var), step=step)
        tf.summary.histogram('histogram', var, step=step)




