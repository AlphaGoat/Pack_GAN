#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("PixelShuffle")
    .Input("unshuffled: float32")
    .Output("shuffled: float32")
    .SetShapeFn([](::
	

