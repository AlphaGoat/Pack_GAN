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
                 #model_scope,
                 layer_scope,
                 summary_update_freq=10):

        # Variables to keep track of name scope of operation
        #self.model_scope = model_scope
        self.layer_scope = layer_scope
        self.batch_norm_name = name
        self.batch_norm_scope =  "{0}/{1}".format(self.layer_scope,
                                                  self.batch_norm_name)

        # Internal variable to keep track of the update frequency for
        # tensorboard summaries
        self.summary_update_freq = summary_update_freq

    def initialize_weights(self, param_shape, layer_scope):
        """
        Initializes tensors to hold mean, variance,
        scale, and bias hyperparameters
        """
#            # Tensor for calculated means (non trainable)
#            self.t_mean = WeightVariable(
#                                shape=(param_shape,),
#                                name='mean_tensor',
#                                model_scope=self.model_scope,
#                                layer_scope=self.batch_norm_scope,
#                                initializer=tf.initializers.zeros,
#                                summary_update_freq=self.summary_update_freq,
#                                )
#
#            # Tensor for calculated variances
#            self.t_var = WeightVariable(
#                                shape=(param_shape,),
#                                name='var_tensor',
#                                model_scope=self.model_scope,
#                                layer_scope=self.batch_norm_scope,
#                                initializer=tf.initializers.zeros,
#                                summary_update_freq=self.summary_update_freq,
#                                )

        # Tensor for offset parameters
        self.t_beta = WeightVariable(
                            shape=(param_shape,),
                            name='offset_tensor',
                            #model_scope=self.model_scope,
                            layer_scope=layer_scope,
                            initializer=tf.zeros,
                            summary_update_freq=self.summary_update_freq,
                            )

        # Tensor for scaling parameters
        self.t_gamma = WeightVariable(
                            shape=(param_shape,),
                            name='scale_tensor',
                            #model_scope=self.model_scope,
                            layer_scope=layer_scope,
                            initializer=tf.zeros,
                            summary_update_freq=self.summary_update_freq,
                            )

    def __call__(self, x, step=0):

        # If first step in training loop, initialize weights
        with tf.name_scope(self.batch_norm_name) as layer_scope:
            if step == 0:
                input_shape = tf.shape(x)
                param_shape = input_shape[-1]

                self.initialize_weights(param_shape, layer_scope)

            # Calculate the mean and variance over batch dimension of input
            mean, variance = tf.nn.moments(x, axes=[0])

            return tf.nn.batch_normalization(x,
                                             mean=mean,
                                             variance=variance,
                                             offset=self.t_beta(step),
                                             scale=self.t_gamma(step),
                                             variance_epsilon=0.001
                                         )

@define_scope
class WeightVariable(object):
    """
    Base class for weight parameters to be used in learning layers
    """
    def __init__(self,
                 shape,
                 name,
                 #model_scope,
                 layer_scope,
                 initializer=None,
                 summary_update_freq=1,
                 ):

        # Shape of initializer parameter tensor
        self.shape = shape

        # Variables to keep track of variable name space
        self.name = name

        # variable to keep track of the scope the variable falls under
        self.layer_scope = layer_scope

        # Update frequency for tensorboard summaries. If 'None',
        # then we are not maintaining a tensorboard for this operation
        self.summary_update_freq = summary_update_freq

        # if no initializer is provided, initialize params as zeros
        if not initializer:
            self.initializer = tf.zeros
        else:
            self.initializer=initializer

        # Internal step variable to pass to summaries
        self._counter = 0

        # Initializer flag. If it is not set, then the
        # variable has not been initialized
        self._initialized = False

    def initialize_variable(self):
        # Initialize Weight Variable
        initial = tf.Variable(
            self.initializer(self.shape),
            trainable=True,
            name=self.name,
            dtype=tf.float32,
            )

        assert initial.name == "%s%s:0" % (self.layer_scope, self.name)

        self.initial = initial

        # Set initialization flag
        self._initialized = True

    #@tf.function
    def __call__(self, step):

        # create summaries for updated weights
        self._counter = step
        if not self._initialized:
            self.initialize_variable()

        if self._counter % self.summary_update_freq == 0:
            variable_summaries(self.initial, self._counter)

        return self.initial

@define_scope
class BiasVariable(object):
    """
    Base class for bias parameters to be used in learning layers
    """
    def __init__(self,
                 shape,
                 name,
                 layer_scope,
                 initializer=tf.initializers.zeros,
                 summary_update_freq=1,
                 ):

        # Variable keeping track of shape of initializer parameter tensor
        self.shape = shape

        # Variables keeping track of variable name space
        self.name = name

        # variable to keep track of the scope the variable falls under
        self.layer_scope = layer_scope

        # If no initializer is provided, initialize params as zeros
        self.initializer = initializer

        # Frequency with which to update tensorboard summaries
        self.summary_update_freq = summary_update_freq

        # Internal step variable to pass to summaries
        self._counter = 0

        # Initializer flag. If it is not set, then the
        # variable has not been initialized
        self._initialized = False

    def initialize_variable(self):
        # Initialize Weight Variable
        #with tf.name_scope(self.layer_scope):
        initial = tf.Variable(
            self.initializer(self.shape),
            trainable=True,
            name=self.name,
            dtype=tf.float32,
            )

        assert initial.name == "%s%s:0" % (self.layer_scope, self.name)

        self.initial = initial

        # set initialization flag
        self._initialized = True

    #@tf.function
    def __call__(self, step):

        # Call variable_summaries for updated weights
        self._counter = step
        if not self._initialized:
            self.initialize_variable()

        if self._counter % self.summary_update_freq == 0:
            variable_summaries(self.initial, self._counter)

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




