Ů8
ŰŹ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéčelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéčelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint˙˙˙˙˙˙˙˙˙
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68×7
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

lstm_4/lstm_cell_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	9**
shared_namelstm_4/lstm_cell_4/kernel

-lstm_4/lstm_cell_4/kernel/Read/ReadVariableOpReadVariableOplstm_4/lstm_cell_4/kernel*
_output_shapes
:	9*
dtype0
¤
#lstm_4/lstm_cell_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#lstm_4/lstm_cell_4/recurrent_kernel

7lstm_4/lstm_cell_4/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_4/lstm_cell_4/recurrent_kernel* 
_output_shapes
:
*
dtype0

lstm_4/lstm_cell_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namelstm_4/lstm_cell_4/bias

+lstm_4/lstm_cell_4/bias/Read/ReadVariableOpReadVariableOplstm_4/lstm_cell_4/bias*
_output_shapes	
:*
dtype0

lstm_5/lstm_cell_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_namelstm_5/lstm_cell_5/kernel

-lstm_5/lstm_cell_5/kernel/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_5/kernel* 
_output_shapes
:
*
dtype0
¤
#lstm_5/lstm_cell_5/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#lstm_5/lstm_cell_5/recurrent_kernel

7lstm_5/lstm_cell_5/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_5/lstm_cell_5/recurrent_kernel* 
_output_shapes
:
*
dtype0

lstm_5/lstm_cell_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namelstm_5/lstm_cell_5/bias

+lstm_5/lstm_cell_5/bias/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_5/bias*
_output_shapes	
:*
dtype0

time_distributed_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	9**
shared_nametime_distributed_2/kernel

-time_distributed_2/kernel/Read/ReadVariableOpReadVariableOptime_distributed_2/kernel*
_output_shapes
:	9*
dtype0

time_distributed_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*(
shared_nametime_distributed_2/bias

+time_distributed_2/bias/Read/ReadVariableOpReadVariableOptime_distributed_2/bias*
_output_shapes
:9*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

 Adam/lstm_4/lstm_cell_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	9*1
shared_name" Adam/lstm_4/lstm_cell_4/kernel/m

4Adam/lstm_4/lstm_cell_4/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_4/lstm_cell_4/kernel/m*
_output_shapes
:	9*
dtype0
˛
*Adam/lstm_4/lstm_cell_4/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*;
shared_name,*Adam/lstm_4/lstm_cell_4/recurrent_kernel/m
Ť
>Adam/lstm_4/lstm_cell_4/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_4/lstm_cell_4/recurrent_kernel/m* 
_output_shapes
:
*
dtype0

Adam/lstm_4/lstm_cell_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/lstm_4/lstm_cell_4/bias/m

2Adam/lstm_4/lstm_cell_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_4/lstm_cell_4/bias/m*
_output_shapes	
:*
dtype0

 Adam/lstm_5/lstm_cell_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Adam/lstm_5/lstm_cell_5/kernel/m

4Adam/lstm_5/lstm_cell_5/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_5/lstm_cell_5/kernel/m* 
_output_shapes
:
*
dtype0
˛
*Adam/lstm_5/lstm_cell_5/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*;
shared_name,*Adam/lstm_5/lstm_cell_5/recurrent_kernel/m
Ť
>Adam/lstm_5/lstm_cell_5/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_5/lstm_cell_5/recurrent_kernel/m* 
_output_shapes
:
*
dtype0

Adam/lstm_5/lstm_cell_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/lstm_5/lstm_cell_5/bias/m

2Adam/lstm_5/lstm_cell_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_5/lstm_cell_5/bias/m*
_output_shapes	
:*
dtype0

 Adam/time_distributed_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	9*1
shared_name" Adam/time_distributed_2/kernel/m

4Adam/time_distributed_2/kernel/m/Read/ReadVariableOpReadVariableOp Adam/time_distributed_2/kernel/m*
_output_shapes
:	9*
dtype0

Adam/time_distributed_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*/
shared_name Adam/time_distributed_2/bias/m

2Adam/time_distributed_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/time_distributed_2/bias/m*
_output_shapes
:9*
dtype0

 Adam/lstm_4/lstm_cell_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	9*1
shared_name" Adam/lstm_4/lstm_cell_4/kernel/v

4Adam/lstm_4/lstm_cell_4/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_4/lstm_cell_4/kernel/v*
_output_shapes
:	9*
dtype0
˛
*Adam/lstm_4/lstm_cell_4/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*;
shared_name,*Adam/lstm_4/lstm_cell_4/recurrent_kernel/v
Ť
>Adam/lstm_4/lstm_cell_4/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_4/lstm_cell_4/recurrent_kernel/v* 
_output_shapes
:
*
dtype0

Adam/lstm_4/lstm_cell_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/lstm_4/lstm_cell_4/bias/v

2Adam/lstm_4/lstm_cell_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_4/lstm_cell_4/bias/v*
_output_shapes	
:*
dtype0

 Adam/lstm_5/lstm_cell_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Adam/lstm_5/lstm_cell_5/kernel/v

4Adam/lstm_5/lstm_cell_5/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_5/lstm_cell_5/kernel/v* 
_output_shapes
:
*
dtype0
˛
*Adam/lstm_5/lstm_cell_5/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*;
shared_name,*Adam/lstm_5/lstm_cell_5/recurrent_kernel/v
Ť
>Adam/lstm_5/lstm_cell_5/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_5/lstm_cell_5/recurrent_kernel/v* 
_output_shapes
:
*
dtype0

Adam/lstm_5/lstm_cell_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/lstm_5/lstm_cell_5/bias/v

2Adam/lstm_5/lstm_cell_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_5/lstm_cell_5/bias/v*
_output_shapes	
:*
dtype0

 Adam/time_distributed_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	9*1
shared_name" Adam/time_distributed_2/kernel/v

4Adam/time_distributed_2/kernel/v/Read/ReadVariableOpReadVariableOp Adam/time_distributed_2/kernel/v*
_output_shapes
:	9*
dtype0

Adam/time_distributed_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*/
shared_name Adam/time_distributed_2/bias/v

2Adam/time_distributed_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/time_distributed_2/bias/v*
_output_shapes
:9*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ž>
value´>Bą> BŞ>
Á
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
Á
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses*
Á
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses*

	layer
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses*
Ř
&iter

'beta_1

(beta_2
	)decay
*learning_rate+mt,mu-mv.mw/mx0my1mz2m{+v|,v}-v~.v/v0v1v2v*
<
+0
,1
-2
.3
/4
05
16
27*
<
+0
,1
-2
.3
/4
05
16
27*
* 
°
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
* 
* 
* 

8serving_default* 
ă
9
state_size

+kernel
,recurrent_kernel
-bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>_random_generator
?__call__
*@&call_and_return_all_conditional_losses*
* 

+0
,1
-2*

+0
,1
-2*
* 


Astates
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
ă
G
state_size

.kernel
/recurrent_kernel
0bias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L_random_generator
M__call__
*N&call_and_return_all_conditional_losses*
* 

.0
/1
02*

.0
/1
02*
* 


Ostates
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
Ś

1kernel
2bias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses*

10
21*

10
21*
* 

[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_4/lstm_cell_4/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#lstm_4/lstm_cell_4/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUElstm_4/lstm_cell_4/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_5/lstm_cell_5/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#lstm_5/lstm_cell_5/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUElstm_5/lstm_cell_5/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEtime_distributed_2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEtime_distributed_2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

`0*
* 
* 
* 
* 

+0
,1
-2*

+0
,1
-2*
* 

anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
:	variables
;trainable_variables
<regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 

.0
/1
02*

.0
/1
02*
* 

fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 

10
21*

10
21*
* 

knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*
* 
* 
* 

0*
* 
* 
* 
8
	ptotal
	qcount
r	variables
s	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

p0
q1*

r	variables*
|v
VARIABLE_VALUE Adam/lstm_4/lstm_cell_4/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/lstm_4/lstm_cell_4/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm_4/lstm_cell_4/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_5/lstm_cell_5/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/lstm_5/lstm_cell_5/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm_5/lstm_cell_5/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/time_distributed_2/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/time_distributed_2/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_4/lstm_cell_4/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/lstm_4/lstm_cell_4/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm_4/lstm_cell_4/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_5/lstm_cell_5/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/lstm_5/lstm_cell_5/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm_5/lstm_cell_5/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/time_distributed_2/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/time_distributed_2/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_lstm_4_inputPlaceholder*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9*
dtype0*)
shape :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
ş
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_4_inputlstm_4/lstm_cell_4/kernellstm_4/lstm_cell_4/bias#lstm_4/lstm_cell_4/recurrent_kernellstm_5/lstm_cell_5/kernellstm_5/lstm_cell_5/bias#lstm_5/lstm_cell_5/recurrent_kerneltime_distributed_2/kerneltime_distributed_2/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_374574
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-lstm_4/lstm_cell_4/kernel/Read/ReadVariableOp7lstm_4/lstm_cell_4/recurrent_kernel/Read/ReadVariableOp+lstm_4/lstm_cell_4/bias/Read/ReadVariableOp-lstm_5/lstm_cell_5/kernel/Read/ReadVariableOp7lstm_5/lstm_cell_5/recurrent_kernel/Read/ReadVariableOp+lstm_5/lstm_cell_5/bias/Read/ReadVariableOp-time_distributed_2/kernel/Read/ReadVariableOp+time_distributed_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp4Adam/lstm_4/lstm_cell_4/kernel/m/Read/ReadVariableOp>Adam/lstm_4/lstm_cell_4/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_4/lstm_cell_4/bias/m/Read/ReadVariableOp4Adam/lstm_5/lstm_cell_5/kernel/m/Read/ReadVariableOp>Adam/lstm_5/lstm_cell_5/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_5/lstm_cell_5/bias/m/Read/ReadVariableOp4Adam/time_distributed_2/kernel/m/Read/ReadVariableOp2Adam/time_distributed_2/bias/m/Read/ReadVariableOp4Adam/lstm_4/lstm_cell_4/kernel/v/Read/ReadVariableOp>Adam/lstm_4/lstm_cell_4/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_4/lstm_cell_4/bias/v/Read/ReadVariableOp4Adam/lstm_5/lstm_cell_5/kernel/v/Read/ReadVariableOp>Adam/lstm_5/lstm_cell_5/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_5/lstm_cell_5/bias/v/Read/ReadVariableOp4Adam/time_distributed_2/kernel/v/Read/ReadVariableOp2Adam/time_distributed_2/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_377380
Ş	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_4/lstm_cell_4/kernel#lstm_4/lstm_cell_4/recurrent_kernellstm_4/lstm_cell_4/biaslstm_5/lstm_cell_5/kernel#lstm_5/lstm_cell_5/recurrent_kernellstm_5/lstm_cell_5/biastime_distributed_2/kerneltime_distributed_2/biastotalcount Adam/lstm_4/lstm_cell_4/kernel/m*Adam/lstm_4/lstm_cell_4/recurrent_kernel/mAdam/lstm_4/lstm_cell_4/bias/m Adam/lstm_5/lstm_cell_5/kernel/m*Adam/lstm_5/lstm_cell_5/recurrent_kernel/mAdam/lstm_5/lstm_cell_5/bias/m Adam/time_distributed_2/kernel/mAdam/time_distributed_2/bias/m Adam/lstm_4/lstm_cell_4/kernel/v*Adam/lstm_4/lstm_cell_4/recurrent_kernel/vAdam/lstm_4/lstm_cell_4/bias/v Adam/lstm_5/lstm_cell_5/kernel/v*Adam/lstm_5/lstm_cell_5/recurrent_kernel/vAdam/lstm_5/lstm_cell_5/bias/v Adam/time_distributed_2/kernel/vAdam/time_distributed_2/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_377483Éë5
š
Ă
while_cond_371524
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_371524___redundant_placeholder04
0while_while_cond_371524___redundant_placeholder14
0while_while_cond_371524___redundant_placeholder24
0while_while_cond_371524___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:
äö
í	
H__inference_sequential_2_layer_call_and_return_conditional_losses_374551

inputsC
0lstm_4_lstm_cell_4_split_readvariableop_resource:	9A
2lstm_4_lstm_cell_4_split_1_readvariableop_resource:	>
*lstm_4_lstm_cell_4_readvariableop_resource:
D
0lstm_5_lstm_cell_5_split_readvariableop_resource:
A
2lstm_5_lstm_cell_5_split_1_readvariableop_resource:	>
*lstm_5_lstm_cell_5_readvariableop_resource:
L
9time_distributed_2_dense_2_matmul_readvariableop_resource:	9H
:time_distributed_2_dense_2_biasadd_readvariableop_resource:9
identity˘!lstm_4/lstm_cell_4/ReadVariableOp˘#lstm_4/lstm_cell_4/ReadVariableOp_1˘#lstm_4/lstm_cell_4/ReadVariableOp_2˘#lstm_4/lstm_cell_4/ReadVariableOp_3˘'lstm_4/lstm_cell_4/split/ReadVariableOp˘)lstm_4/lstm_cell_4/split_1/ReadVariableOp˘lstm_4/while˘!lstm_5/lstm_cell_5/ReadVariableOp˘#lstm_5/lstm_cell_5/ReadVariableOp_1˘#lstm_5/lstm_cell_5/ReadVariableOp_2˘#lstm_5/lstm_cell_5/ReadVariableOp_3˘'lstm_5/lstm_cell_5/split/ReadVariableOp˘)lstm_5/lstm_cell_5/split_1/ReadVariableOp˘lstm_5/while˘1time_distributed_2/dense_2/BiasAdd/ReadVariableOp˘0time_distributed_2/dense_2/MatMul/ReadVariableOpB
lstm_4/ShapeShapeinputs*
T0*
_output_shapes
:d
lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_4/strided_sliceStridedSlicelstm_4/Shape:output:0#lstm_4/strided_slice/stack:output:0%lstm_4/strided_slice/stack_1:output:0%lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_4/zeros/packedPacklstm_4/strided_slice:output:0lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_4/zerosFilllstm_4/zeros/packed:output:0lstm_4/zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_4/zeros_1/packedPacklstm_4/strided_slice:output:0 lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_4/zeros_1Filllstm_4/zeros_1/packed:output:0lstm_4/zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_4/transpose	Transposeinputslstm_4/transpose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9R
lstm_4/Shape_1Shapelstm_4/transpose:y:0*
T0*
_output_shapes
:f
lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ţ
lstm_4/strided_slice_1StridedSlicelstm_4/Shape_1:output:0%lstm_4/strided_slice_1/stack:output:0'lstm_4/strided_slice_1/stack_1:output:0'lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙É
lstm_4/TensorArrayV2TensorListReserve+lstm_4/TensorArrayV2/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
<lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   ő
.lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_4/transpose:y:0Elstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇf
lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_4/strided_slice_2StridedSlicelstm_4/transpose:y:0%lstm_4/strided_slice_2/stack:output:0'lstm_4/strided_slice_2/stack_1:output:0'lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
shrink_axis_maskg
"lstm_4/lstm_cell_4/ones_like/ShapeShapelstm_4/zeros:output:0*
T0*
_output_shapes
:g
"lstm_4/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ą
lstm_4/lstm_cell_4/ones_likeFill+lstm_4/lstm_cell_4/ones_like/Shape:output:0+lstm_4/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
 lstm_4/lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ş
lstm_4/lstm_cell_4/dropout/MulMul%lstm_4/lstm_cell_4/ones_like:output:0)lstm_4/lstm_cell_4/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
 lstm_4/lstm_cell_4/dropout/ShapeShape%lstm_4/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:ł
7lstm_4/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform)lstm_4/lstm_cell_4/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0n
)lstm_4/lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ŕ
'lstm_4/lstm_cell_4/dropout/GreaterEqualGreaterEqual@lstm_4/lstm_cell_4/dropout/random_uniform/RandomUniform:output:02lstm_4/lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/dropout/CastCast+lstm_4/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
 lstm_4/lstm_cell_4/dropout/Mul_1Mul"lstm_4/lstm_cell_4/dropout/Mul:z:0#lstm_4/lstm_cell_4/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
"lstm_4/lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ž
 lstm_4/lstm_cell_4/dropout_1/MulMul%lstm_4/lstm_cell_4/ones_like:output:0+lstm_4/lstm_cell_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
"lstm_4/lstm_cell_4/dropout_1/ShapeShape%lstm_4/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:ˇ
9lstm_4/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform+lstm_4/lstm_cell_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0p
+lstm_4/lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ć
)lstm_4/lstm_cell_4/dropout_1/GreaterEqualGreaterEqualBlstm_4/lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:04lstm_4/lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!lstm_4/lstm_cell_4/dropout_1/CastCast-lstm_4/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Š
"lstm_4/lstm_cell_4/dropout_1/Mul_1Mul$lstm_4/lstm_cell_4/dropout_1/Mul:z:0%lstm_4/lstm_cell_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
"lstm_4/lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ž
 lstm_4/lstm_cell_4/dropout_2/MulMul%lstm_4/lstm_cell_4/ones_like:output:0+lstm_4/lstm_cell_4/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
"lstm_4/lstm_cell_4/dropout_2/ShapeShape%lstm_4/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:ˇ
9lstm_4/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform+lstm_4/lstm_cell_4/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0p
+lstm_4/lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ć
)lstm_4/lstm_cell_4/dropout_2/GreaterEqualGreaterEqualBlstm_4/lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:04lstm_4/lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!lstm_4/lstm_cell_4/dropout_2/CastCast-lstm_4/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Š
"lstm_4/lstm_cell_4/dropout_2/Mul_1Mul$lstm_4/lstm_cell_4/dropout_2/Mul:z:0%lstm_4/lstm_cell_4/dropout_2/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
"lstm_4/lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ž
 lstm_4/lstm_cell_4/dropout_3/MulMul%lstm_4/lstm_cell_4/ones_like:output:0+lstm_4/lstm_cell_4/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
"lstm_4/lstm_cell_4/dropout_3/ShapeShape%lstm_4/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:ˇ
9lstm_4/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform+lstm_4/lstm_cell_4/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0p
+lstm_4/lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ć
)lstm_4/lstm_cell_4/dropout_3/GreaterEqualGreaterEqualBlstm_4/lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:04lstm_4/lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!lstm_4/lstm_cell_4/dropout_3/CastCast-lstm_4/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Š
"lstm_4/lstm_cell_4/dropout_3/Mul_1Mul$lstm_4/lstm_cell_4/dropout_3/Mul:z:0%lstm_4/lstm_cell_4/dropout_3/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
"lstm_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'lstm_4/lstm_cell_4/split/ReadVariableOpReadVariableOp0lstm_4_lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	9*
dtype0Ű
lstm_4/lstm_cell_4/splitSplit+lstm_4/lstm_cell_4/split/split_dim:output:0/lstm_4/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_split
lstm_4/lstm_cell_4/MatMulMatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/MatMul_1MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/MatMul_2MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/MatMul_3MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
$lstm_4/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
)lstm_4/lstm_cell_4/split_1/ReadVariableOpReadVariableOp2lstm_4_lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0Ń
lstm_4/lstm_cell_4/split_1Split-lstm_4/lstm_cell_4/split_1/split_dim:output:01lstm_4/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split˘
lstm_4/lstm_cell_4/BiasAddBiasAdd#lstm_4/lstm_cell_4/MatMul:product:0#lstm_4/lstm_cell_4/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
lstm_4/lstm_cell_4/BiasAdd_1BiasAdd%lstm_4/lstm_cell_4/MatMul_1:product:0#lstm_4/lstm_cell_4/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
lstm_4/lstm_cell_4/BiasAdd_2BiasAdd%lstm_4/lstm_cell_4/MatMul_2:product:0#lstm_4/lstm_cell_4/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
lstm_4/lstm_cell_4/BiasAdd_3BiasAdd%lstm_4/lstm_cell_4/MatMul_3:product:0#lstm_4/lstm_cell_4/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/mulMullstm_4/zeros:output:0$lstm_4/lstm_cell_4/dropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/mul_1Mullstm_4/zeros:output:0&lstm_4/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/mul_2Mullstm_4/zeros:output:0&lstm_4/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/mul_3Mullstm_4/zeros:output:0&lstm_4/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!lstm_4/lstm_cell_4/ReadVariableOpReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0w
&lstm_4/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(lstm_4/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(lstm_4/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 lstm_4/lstm_cell_4/strided_sliceStridedSlice)lstm_4/lstm_cell_4/ReadVariableOp:value:0/lstm_4/lstm_cell_4/strided_slice/stack:output:01lstm_4/lstm_cell_4/strided_slice/stack_1:output:01lstm_4/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_4/lstm_cell_4/MatMul_4MatMullstm_4/lstm_cell_4/mul:z:0)lstm_4/lstm_cell_4/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/addAddV2#lstm_4/lstm_cell_4/BiasAdd:output:0%lstm_4/lstm_cell_4/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lstm_4/lstm_cell_4/SigmoidSigmoidlstm_4/lstm_cell_4/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#lstm_4/lstm_cell_4/ReadVariableOp_1ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0y
(lstm_4/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_4/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_4/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm_4/lstm_cell_4/strided_slice_1StridedSlice+lstm_4/lstm_cell_4/ReadVariableOp_1:value:01lstm_4/lstm_cell_4/strided_slice_1/stack:output:03lstm_4/lstm_cell_4/strided_slice_1/stack_1:output:03lstm_4/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŁ
lstm_4/lstm_cell_4/MatMul_5MatMullstm_4/lstm_cell_4/mul_1:z:0+lstm_4/lstm_cell_4/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_4/lstm_cell_4/add_1AddV2%lstm_4/lstm_cell_4/BiasAdd_1:output:0%lstm_4/lstm_cell_4/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_4/lstm_cell_4/Sigmoid_1Sigmoidlstm_4/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/mul_4Mul lstm_4/lstm_cell_4/Sigmoid_1:y:0lstm_4/zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#lstm_4/lstm_cell_4/ReadVariableOp_2ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0y
(lstm_4/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_4/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      {
*lstm_4/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm_4/lstm_cell_4/strided_slice_2StridedSlice+lstm_4/lstm_cell_4/ReadVariableOp_2:value:01lstm_4/lstm_cell_4/strided_slice_2/stack:output:03lstm_4/lstm_cell_4/strided_slice_2/stack_1:output:03lstm_4/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŁ
lstm_4/lstm_cell_4/MatMul_6MatMullstm_4/lstm_cell_4/mul_2:z:0+lstm_4/lstm_cell_4/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_4/lstm_cell_4/add_2AddV2%lstm_4/lstm_cell_4/BiasAdd_2:output:0%lstm_4/lstm_cell_4/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
lstm_4/lstm_cell_4/TanhTanhlstm_4/lstm_cell_4/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/mul_5Mullstm_4/lstm_cell_4/Sigmoid:y:0lstm_4/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/add_3AddV2lstm_4/lstm_cell_4/mul_4:z:0lstm_4/lstm_cell_4/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#lstm_4/lstm_cell_4/ReadVariableOp_3ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0y
(lstm_4/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      {
*lstm_4/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_4/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm_4/lstm_cell_4/strided_slice_3StridedSlice+lstm_4/lstm_cell_4/ReadVariableOp_3:value:01lstm_4/lstm_cell_4/strided_slice_3/stack:output:03lstm_4/lstm_cell_4/strided_slice_3/stack_1:output:03lstm_4/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŁ
lstm_4/lstm_cell_4/MatMul_7MatMullstm_4/lstm_cell_4/mul_3:z:0+lstm_4/lstm_cell_4/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_4/lstm_cell_4/add_4AddV2%lstm_4/lstm_cell_4/BiasAdd_3:output:0%lstm_4/lstm_cell_4/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_4/lstm_cell_4/Sigmoid_2Sigmoidlstm_4/lstm_cell_4/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
lstm_4/lstm_cell_4/Tanh_1Tanhlstm_4/lstm_cell_4/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/mul_6Mul lstm_4/lstm_cell_4/Sigmoid_2:y:0lstm_4/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
$lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Í
lstm_4/TensorArrayV2_1TensorListReserve-lstm_4/TensorArrayV2_1/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇM
lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙[
lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ű
lstm_4/whileWhile"lstm_4/while/loop_counter:output:0(lstm_4/while/maximum_iterations:output:0lstm_4/time:output:0lstm_4/TensorArrayV2_1:handle:0lstm_4/zeros:output:0lstm_4/zeros_1:output:0lstm_4/strided_slice_1:output:0>lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_4_lstm_cell_4_split_readvariableop_resource2lstm_4_lstm_cell_4_split_1_readvariableop_resource*lstm_4_lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_4_while_body_374083*$
condR
lstm_4_while_cond_374082*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
7lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   á
)lstm_4/TensorArrayV2Stack/TensorListStackTensorListStacklstm_4/while:output:3@lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0o
lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙h
lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ť
lstm_4/strided_slice_3StridedSlice2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_4/strided_slice_3/stack:output:0'lstm_4/strided_slice_3/stack_1:output:0'lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskl
lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ľ
lstm_4/transpose_1	Transpose2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_4/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙b
lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    R
lstm_5/ShapeShapelstm_4/transpose_1:y:0*
T0*
_output_shapes
:d
lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_5/strided_sliceStridedSlicelstm_5/Shape:output:0#lstm_5/strided_slice/stack:output:0%lstm_5/strided_slice/stack_1:output:0%lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_5/zeros/packedPacklstm_5/strided_slice:output:0lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_5/zerosFilllstm_5/zeros/packed:output:0lstm_5/zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_5/zeros_1/packedPacklstm_5/strided_slice:output:0 lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_5/zeros_1Filllstm_5/zeros_1/packed:output:0lstm_5/zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_5/transpose	Transposelstm_4/transpose_1:y:0lstm_5/transpose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙R
lstm_5/Shape_1Shapelstm_5/transpose:y:0*
T0*
_output_shapes
:f
lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ţ
lstm_5/strided_slice_1StridedSlicelstm_5/Shape_1:output:0%lstm_5/strided_slice_1/stack:output:0'lstm_5/strided_slice_1/stack_1:output:0'lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙É
lstm_5/TensorArrayV2TensorListReserve+lstm_5/TensorArrayV2/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ő
.lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_5/transpose:y:0Elstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇf
lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_5/strided_slice_2StridedSlicelstm_5/transpose:y:0%lstm_5/strided_slice_2/stack:output:0'lstm_5/strided_slice_2/stack_1:output:0'lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskg
"lstm_5/lstm_cell_5/ones_like/ShapeShapelstm_5/zeros:output:0*
T0*
_output_shapes
:g
"lstm_5/lstm_cell_5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ą
lstm_5/lstm_cell_5/ones_likeFill+lstm_5/lstm_cell_5/ones_like/Shape:output:0+lstm_5/lstm_cell_5/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
 lstm_5/lstm_cell_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ş
lstm_5/lstm_cell_5/dropout/MulMul%lstm_5/lstm_cell_5/ones_like:output:0)lstm_5/lstm_cell_5/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
 lstm_5/lstm_cell_5/dropout/ShapeShape%lstm_5/lstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:ł
7lstm_5/lstm_cell_5/dropout/random_uniform/RandomUniformRandomUniform)lstm_5/lstm_cell_5/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0n
)lstm_5/lstm_cell_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ŕ
'lstm_5/lstm_cell_5/dropout/GreaterEqualGreaterEqual@lstm_5/lstm_cell_5/dropout/random_uniform/RandomUniform:output:02lstm_5/lstm_cell_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/dropout/CastCast+lstm_5/lstm_cell_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
 lstm_5/lstm_cell_5/dropout/Mul_1Mul"lstm_5/lstm_cell_5/dropout/Mul:z:0#lstm_5/lstm_cell_5/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
"lstm_5/lstm_cell_5/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ž
 lstm_5/lstm_cell_5/dropout_1/MulMul%lstm_5/lstm_cell_5/ones_like:output:0+lstm_5/lstm_cell_5/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
"lstm_5/lstm_cell_5/dropout_1/ShapeShape%lstm_5/lstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:ˇ
9lstm_5/lstm_cell_5/dropout_1/random_uniform/RandomUniformRandomUniform+lstm_5/lstm_cell_5/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0p
+lstm_5/lstm_cell_5/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ć
)lstm_5/lstm_cell_5/dropout_1/GreaterEqualGreaterEqualBlstm_5/lstm_cell_5/dropout_1/random_uniform/RandomUniform:output:04lstm_5/lstm_cell_5/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!lstm_5/lstm_cell_5/dropout_1/CastCast-lstm_5/lstm_cell_5/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Š
"lstm_5/lstm_cell_5/dropout_1/Mul_1Mul$lstm_5/lstm_cell_5/dropout_1/Mul:z:0%lstm_5/lstm_cell_5/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
"lstm_5/lstm_cell_5/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ž
 lstm_5/lstm_cell_5/dropout_2/MulMul%lstm_5/lstm_cell_5/ones_like:output:0+lstm_5/lstm_cell_5/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
"lstm_5/lstm_cell_5/dropout_2/ShapeShape%lstm_5/lstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:ˇ
9lstm_5/lstm_cell_5/dropout_2/random_uniform/RandomUniformRandomUniform+lstm_5/lstm_cell_5/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0p
+lstm_5/lstm_cell_5/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ć
)lstm_5/lstm_cell_5/dropout_2/GreaterEqualGreaterEqualBlstm_5/lstm_cell_5/dropout_2/random_uniform/RandomUniform:output:04lstm_5/lstm_cell_5/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!lstm_5/lstm_cell_5/dropout_2/CastCast-lstm_5/lstm_cell_5/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Š
"lstm_5/lstm_cell_5/dropout_2/Mul_1Mul$lstm_5/lstm_cell_5/dropout_2/Mul:z:0%lstm_5/lstm_cell_5/dropout_2/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
"lstm_5/lstm_cell_5/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ž
 lstm_5/lstm_cell_5/dropout_3/MulMul%lstm_5/lstm_cell_5/ones_like:output:0+lstm_5/lstm_cell_5/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
"lstm_5/lstm_cell_5/dropout_3/ShapeShape%lstm_5/lstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:ˇ
9lstm_5/lstm_cell_5/dropout_3/random_uniform/RandomUniformRandomUniform+lstm_5/lstm_cell_5/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0p
+lstm_5/lstm_cell_5/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ć
)lstm_5/lstm_cell_5/dropout_3/GreaterEqualGreaterEqualBlstm_5/lstm_cell_5/dropout_3/random_uniform/RandomUniform:output:04lstm_5/lstm_cell_5/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!lstm_5/lstm_cell_5/dropout_3/CastCast-lstm_5/lstm_cell_5/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Š
"lstm_5/lstm_cell_5/dropout_3/Mul_1Mul$lstm_5/lstm_cell_5/dropout_3/Mul:z:0%lstm_5/lstm_cell_5/dropout_3/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
"lstm_5/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'lstm_5/lstm_cell_5/split/ReadVariableOpReadVariableOp0lstm_5_lstm_cell_5_split_readvariableop_resource* 
_output_shapes
:
*
dtype0ß
lstm_5/lstm_cell_5/splitSplit+lstm_5/lstm_cell_5/split/split_dim:output:0/lstm_5/lstm_cell_5/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split
lstm_5/lstm_cell_5/MatMulMatMullstm_5/strided_slice_2:output:0!lstm_5/lstm_cell_5/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/MatMul_1MatMullstm_5/strided_slice_2:output:0!lstm_5/lstm_cell_5/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/MatMul_2MatMullstm_5/strided_slice_2:output:0!lstm_5/lstm_cell_5/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/MatMul_3MatMullstm_5/strided_slice_2:output:0!lstm_5/lstm_cell_5/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
$lstm_5/lstm_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
)lstm_5/lstm_cell_5/split_1/ReadVariableOpReadVariableOp2lstm_5_lstm_cell_5_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0Ń
lstm_5/lstm_cell_5/split_1Split-lstm_5/lstm_cell_5/split_1/split_dim:output:01lstm_5/lstm_cell_5/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split˘
lstm_5/lstm_cell_5/BiasAddBiasAdd#lstm_5/lstm_cell_5/MatMul:product:0#lstm_5/lstm_cell_5/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
lstm_5/lstm_cell_5/BiasAdd_1BiasAdd%lstm_5/lstm_cell_5/MatMul_1:product:0#lstm_5/lstm_cell_5/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
lstm_5/lstm_cell_5/BiasAdd_2BiasAdd%lstm_5/lstm_cell_5/MatMul_2:product:0#lstm_5/lstm_cell_5/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
lstm_5/lstm_cell_5/BiasAdd_3BiasAdd%lstm_5/lstm_cell_5/MatMul_3:product:0#lstm_5/lstm_cell_5/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/mulMullstm_5/zeros:output:0$lstm_5/lstm_cell_5/dropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/mul_1Mullstm_5/zeros:output:0&lstm_5/lstm_cell_5/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/mul_2Mullstm_5/zeros:output:0&lstm_5/lstm_cell_5/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/mul_3Mullstm_5/zeros:output:0&lstm_5/lstm_cell_5/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!lstm_5/lstm_cell_5/ReadVariableOpReadVariableOp*lstm_5_lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0w
&lstm_5/lstm_cell_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(lstm_5/lstm_cell_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(lstm_5/lstm_cell_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 lstm_5/lstm_cell_5/strided_sliceStridedSlice)lstm_5/lstm_cell_5/ReadVariableOp:value:0/lstm_5/lstm_cell_5/strided_slice/stack:output:01lstm_5/lstm_cell_5/strided_slice/stack_1:output:01lstm_5/lstm_cell_5/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_5/lstm_cell_5/MatMul_4MatMullstm_5/lstm_cell_5/mul:z:0)lstm_5/lstm_cell_5/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/addAddV2#lstm_5/lstm_cell_5/BiasAdd:output:0%lstm_5/lstm_cell_5/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lstm_5/lstm_cell_5/SigmoidSigmoidlstm_5/lstm_cell_5/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#lstm_5/lstm_cell_5/ReadVariableOp_1ReadVariableOp*lstm_5_lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0y
(lstm_5/lstm_cell_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_5/lstm_cell_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_5/lstm_cell_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm_5/lstm_cell_5/strided_slice_1StridedSlice+lstm_5/lstm_cell_5/ReadVariableOp_1:value:01lstm_5/lstm_cell_5/strided_slice_1/stack:output:03lstm_5/lstm_cell_5/strided_slice_1/stack_1:output:03lstm_5/lstm_cell_5/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŁ
lstm_5/lstm_cell_5/MatMul_5MatMullstm_5/lstm_cell_5/mul_1:z:0+lstm_5/lstm_cell_5/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_5/lstm_cell_5/add_1AddV2%lstm_5/lstm_cell_5/BiasAdd_1:output:0%lstm_5/lstm_cell_5/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_5/lstm_cell_5/Sigmoid_1Sigmoidlstm_5/lstm_cell_5/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/mul_4Mul lstm_5/lstm_cell_5/Sigmoid_1:y:0lstm_5/zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#lstm_5/lstm_cell_5/ReadVariableOp_2ReadVariableOp*lstm_5_lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0y
(lstm_5/lstm_cell_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_5/lstm_cell_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      {
*lstm_5/lstm_cell_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm_5/lstm_cell_5/strided_slice_2StridedSlice+lstm_5/lstm_cell_5/ReadVariableOp_2:value:01lstm_5/lstm_cell_5/strided_slice_2/stack:output:03lstm_5/lstm_cell_5/strided_slice_2/stack_1:output:03lstm_5/lstm_cell_5/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŁ
lstm_5/lstm_cell_5/MatMul_6MatMullstm_5/lstm_cell_5/mul_2:z:0+lstm_5/lstm_cell_5/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_5/lstm_cell_5/add_2AddV2%lstm_5/lstm_cell_5/BiasAdd_2:output:0%lstm_5/lstm_cell_5/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
lstm_5/lstm_cell_5/TanhTanhlstm_5/lstm_cell_5/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/mul_5Mullstm_5/lstm_cell_5/Sigmoid:y:0lstm_5/lstm_cell_5/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/add_3AddV2lstm_5/lstm_cell_5/mul_4:z:0lstm_5/lstm_cell_5/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#lstm_5/lstm_cell_5/ReadVariableOp_3ReadVariableOp*lstm_5_lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0y
(lstm_5/lstm_cell_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      {
*lstm_5/lstm_cell_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_5/lstm_cell_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm_5/lstm_cell_5/strided_slice_3StridedSlice+lstm_5/lstm_cell_5/ReadVariableOp_3:value:01lstm_5/lstm_cell_5/strided_slice_3/stack:output:03lstm_5/lstm_cell_5/strided_slice_3/stack_1:output:03lstm_5/lstm_cell_5/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŁ
lstm_5/lstm_cell_5/MatMul_7MatMullstm_5/lstm_cell_5/mul_3:z:0+lstm_5/lstm_cell_5/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_5/lstm_cell_5/add_4AddV2%lstm_5/lstm_cell_5/BiasAdd_3:output:0%lstm_5/lstm_cell_5/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_5/lstm_cell_5/Sigmoid_2Sigmoidlstm_5/lstm_cell_5/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
lstm_5/lstm_cell_5/Tanh_1Tanhlstm_5/lstm_cell_5/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/mul_6Mul lstm_5/lstm_cell_5/Sigmoid_2:y:0lstm_5/lstm_cell_5/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
$lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Í
lstm_5/TensorArrayV2_1TensorListReserve-lstm_5/TensorArrayV2_1/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇM
lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙[
lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ű
lstm_5/whileWhile"lstm_5/while/loop_counter:output:0(lstm_5/while/maximum_iterations:output:0lstm_5/time:output:0lstm_5/TensorArrayV2_1:handle:0lstm_5/zeros:output:0lstm_5/zeros_1:output:0lstm_5/strided_slice_1:output:0>lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_5_lstm_cell_5_split_readvariableop_resource2lstm_5_lstm_cell_5_split_1_readvariableop_resource*lstm_5_lstm_cell_5_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_5_while_body_374372*$
condR
lstm_5_while_cond_374371*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   á
)lstm_5/TensorArrayV2Stack/TensorListStackTensorListStacklstm_5/while:output:3@lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0o
lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙h
lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ť
lstm_5/strided_slice_3StridedSlice2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_5/strided_slice_3/stack:output:0'lstm_5/strided_slice_3/stack_1:output:0'lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskl
lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ľ
lstm_5/transpose_1	Transpose2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_5/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙b
lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ^
time_distributed_2/ShapeShapelstm_5/transpose_1:y:0*
T0*
_output_shapes
:p
&time_distributed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(time_distributed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(time_distributed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 time_distributed_2/strided_sliceStridedSlice!time_distributed_2/Shape:output:0/time_distributed_2/strided_slice/stack:output:01time_distributed_2/strided_slice/stack_1:output:01time_distributed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
time_distributed_2/ReshapeReshapelstm_5/transpose_1:y:0)time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
0time_distributed_2/dense_2/MatMul/ReadVariableOpReadVariableOp9time_distributed_2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	9*
dtype0ź
!time_distributed_2/dense_2/MatMulMatMul#time_distributed_2/Reshape:output:08time_distributed_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9¨
1time_distributed_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp:time_distributed_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:9*
dtype0Ç
"time_distributed_2/dense_2/BiasAddBiasAdd+time_distributed_2/dense_2/MatMul:product:09time_distributed_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9
"time_distributed_2/dense_2/SoftmaxSoftmax+time_distributed_2/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9o
$time_distributed_2/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙f
$time_distributed_2/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :9á
"time_distributed_2/Reshape_1/shapePack-time_distributed_2/Reshape_1/shape/0:output:0)time_distributed_2/strided_slice:output:0-time_distributed_2/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:Á
time_distributed_2/Reshape_1Reshape,time_distributed_2/dense_2/Softmax:softmax:0+time_distributed_2/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9s
"time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
time_distributed_2/Reshape_2Reshapelstm_5/transpose_1:y:0+time_distributed_2/Reshape_2/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
IdentityIdentity%time_distributed_2/Reshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9Ł
NoOpNoOp"^lstm_4/lstm_cell_4/ReadVariableOp$^lstm_4/lstm_cell_4/ReadVariableOp_1$^lstm_4/lstm_cell_4/ReadVariableOp_2$^lstm_4/lstm_cell_4/ReadVariableOp_3(^lstm_4/lstm_cell_4/split/ReadVariableOp*^lstm_4/lstm_cell_4/split_1/ReadVariableOp^lstm_4/while"^lstm_5/lstm_cell_5/ReadVariableOp$^lstm_5/lstm_cell_5/ReadVariableOp_1$^lstm_5/lstm_cell_5/ReadVariableOp_2$^lstm_5/lstm_cell_5/ReadVariableOp_3(^lstm_5/lstm_cell_5/split/ReadVariableOp*^lstm_5/lstm_cell_5/split_1/ReadVariableOp^lstm_5/while2^time_distributed_2/dense_2/BiasAdd/ReadVariableOp1^time_distributed_2/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : : : : : : 2F
!lstm_4/lstm_cell_4/ReadVariableOp!lstm_4/lstm_cell_4/ReadVariableOp2J
#lstm_4/lstm_cell_4/ReadVariableOp_1#lstm_4/lstm_cell_4/ReadVariableOp_12J
#lstm_4/lstm_cell_4/ReadVariableOp_2#lstm_4/lstm_cell_4/ReadVariableOp_22J
#lstm_4/lstm_cell_4/ReadVariableOp_3#lstm_4/lstm_cell_4/ReadVariableOp_32R
'lstm_4/lstm_cell_4/split/ReadVariableOp'lstm_4/lstm_cell_4/split/ReadVariableOp2V
)lstm_4/lstm_cell_4/split_1/ReadVariableOp)lstm_4/lstm_cell_4/split_1/ReadVariableOp2
lstm_4/whilelstm_4/while2F
!lstm_5/lstm_cell_5/ReadVariableOp!lstm_5/lstm_cell_5/ReadVariableOp2J
#lstm_5/lstm_cell_5/ReadVariableOp_1#lstm_5/lstm_cell_5/ReadVariableOp_12J
#lstm_5/lstm_cell_5/ReadVariableOp_2#lstm_5/lstm_cell_5/ReadVariableOp_22J
#lstm_5/lstm_cell_5/ReadVariableOp_3#lstm_5/lstm_cell_5/ReadVariableOp_32R
'lstm_5/lstm_cell_5/split/ReadVariableOp'lstm_5/lstm_cell_5/split/ReadVariableOp2V
)lstm_5/lstm_cell_5/split_1/ReadVariableOp)lstm_5/lstm_cell_5/split_1/ReadVariableOp2
lstm_5/whilelstm_5/while2f
1time_distributed_2/dense_2/BiasAdd/ReadVariableOp1time_distributed_2/dense_2/BiasAdd/ReadVariableOp2d
0time_distributed_2/dense_2/MatMul/ReadVariableOp0time_distributed_2/dense_2/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
 
_user_specified_nameinputs
§
ś
'__inference_lstm_5_layer_call_fn_375706

inputs
unknown:

	unknown_0:	
	unknown_1:

identity˘StatefulPartitionedCallň
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_372965}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ńl
	
while_body_374720
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_4_split_readvariableop_resource_0:	9B
3while_lstm_cell_4_split_1_readvariableop_resource_0:	?
+while_lstm_cell_4_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_4_split_readvariableop_resource:	9@
1while_lstm_cell_4_split_1_readvariableop_resource:	=
)while_lstm_cell_4_readvariableop_resource:
˘ while/lstm_cell_4/ReadVariableOp˘"while/lstm_cell_4/ReadVariableOp_1˘"while/lstm_cell_4/ReadVariableOp_2˘"while/lstm_cell_4/ReadVariableOp_3˘&while/lstm_cell_4/split/ReadVariableOp˘(while/lstm_cell_4/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
element_dtype0d
!while/lstm_cell_4/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ž
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	9*
dtype0Ř
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_splitŠ
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Î
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mulMulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_1Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_2Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_3Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
while/lstm_cell_4/TanhTanhwhile/lstm_cell_4/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0while/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/lstm_cell_4/Tanh_1Tanhwhile/lstm_cell_4/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0while/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇy
while/Identity_4Identitywhile/lstm_cell_4/mul_6:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˛

while/NoOpNoOp!^while/lstm_cell_4/ReadVariableOp#^while/lstm_cell_4/ReadVariableOp_1#^while/lstm_cell_4/ReadVariableOp_2#^while/lstm_cell_4/ReadVariableOp_3'^while/lstm_cell_4/split/ReadVariableOp)^while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2D
 while/lstm_cell_4/ReadVariableOp while/lstm_cell_4/ReadVariableOp2H
"while/lstm_cell_4/ReadVariableOp_1"while/lstm_cell_4/ReadVariableOp_12H
"while/lstm_cell_4/ReadVariableOp_2"while/lstm_cell_4/ReadVariableOp_22H
"while/lstm_cell_4/ReadVariableOp_3"while/lstm_cell_4/ReadVariableOp_32P
&while/lstm_cell_4/split/ReadVariableOp&while/lstm_cell_4/split/ReadVariableOp2T
(while/lstm_cell_4/split_1/ReadVariableOp(while/lstm_cell_4/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 
ę
Ó
%sequential_2_lstm_5_while_cond_370987D
@sequential_2_lstm_5_while_sequential_2_lstm_5_while_loop_counterJ
Fsequential_2_lstm_5_while_sequential_2_lstm_5_while_maximum_iterations)
%sequential_2_lstm_5_while_placeholder+
'sequential_2_lstm_5_while_placeholder_1+
'sequential_2_lstm_5_while_placeholder_2+
'sequential_2_lstm_5_while_placeholder_3F
Bsequential_2_lstm_5_while_less_sequential_2_lstm_5_strided_slice_1\
Xsequential_2_lstm_5_while_sequential_2_lstm_5_while_cond_370987___redundant_placeholder0\
Xsequential_2_lstm_5_while_sequential_2_lstm_5_while_cond_370987___redundant_placeholder1\
Xsequential_2_lstm_5_while_sequential_2_lstm_5_while_cond_370987___redundant_placeholder2\
Xsequential_2_lstm_5_while_sequential_2_lstm_5_while_cond_370987___redundant_placeholder3&
"sequential_2_lstm_5_while_identity
˛
sequential_2/lstm_5/while/LessLess%sequential_2_lstm_5_while_placeholderBsequential_2_lstm_5_while_less_sequential_2_lstm_5_strided_slice_1*
T0*
_output_shapes
: s
"sequential_2/lstm_5/while/IdentityIdentity"sequential_2/lstm_5/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_2_lstm_5_while_identity+sequential_2/lstm_5/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:
š
Ă
while_cond_376329
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_376329___redundant_placeholder04
0while_while_cond_376329___redundant_placeholder14
0while_while_cond_376329___redundant_placeholder24
0while_while_cond_376329___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:
Ü
ß
B__inference_lstm_5_layer_call_and_return_conditional_losses_372965

inputs=
)lstm_cell_5_split_readvariableop_resource:
:
+lstm_cell_5_split_1_readvariableop_resource:	7
#lstm_cell_5_readvariableop_resource:

identity˘lstm_cell_5/ReadVariableOp˘lstm_cell_5/ReadVariableOp_1˘lstm_cell_5/ReadVariableOp_2˘lstm_cell_5/ReadVariableOp_3˘ lstm_cell_5/split/ReadVariableOp˘"lstm_cell_5/split_1/ReadVariableOp˘while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ę
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskY
lstm_cell_5/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell_5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_5/ones_likeFill$lstm_cell_5/ones_like/Shape:output:0$lstm_cell_5/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_5/dropout/MulMullstm_cell_5/ones_like:output:0"lstm_cell_5/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
lstm_cell_5/dropout/ShapeShapelstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:Ľ
0lstm_cell_5/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_5/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ë
 lstm_cell_5/dropout/GreaterEqualGreaterEqual9lstm_cell_5/dropout/random_uniform/RandomUniform:output:0+lstm_cell_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout/CastCast$lstm_cell_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout/Mul_1Mullstm_cell_5/dropout/Mul:z:0lstm_cell_5/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_5/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_5/dropout_1/MulMullstm_cell_5/ones_like:output:0$lstm_cell_5/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
lstm_cell_5/dropout_1/ShapeShapelstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:Š
2lstm_cell_5/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_5/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0i
$lstm_cell_5/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ń
"lstm_cell_5/dropout_1/GreaterEqualGreaterEqual;lstm_cell_5/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_5/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout_1/CastCast&lstm_cell_5/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout_1/Mul_1Mullstm_cell_5/dropout_1/Mul:z:0lstm_cell_5/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_5/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_5/dropout_2/MulMullstm_cell_5/ones_like:output:0$lstm_cell_5/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
lstm_cell_5/dropout_2/ShapeShapelstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:Š
2lstm_cell_5/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_5/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0i
$lstm_cell_5/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ń
"lstm_cell_5/dropout_2/GreaterEqualGreaterEqual;lstm_cell_5/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_5/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout_2/CastCast&lstm_cell_5/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout_2/Mul_1Mullstm_cell_5/dropout_2/Mul:z:0lstm_cell_5/dropout_2/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_5/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_5/dropout_3/MulMullstm_cell_5/ones_like:output:0$lstm_cell_5/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
lstm_cell_5/dropout_3/ShapeShapelstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:Š
2lstm_cell_5/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_5/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0i
$lstm_cell_5/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ń
"lstm_cell_5/dropout_3/GreaterEqualGreaterEqual;lstm_cell_5/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_5/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout_3/CastCast&lstm_cell_5/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout_3/Mul_1Mullstm_cell_5/dropout_3/Mul:z:0lstm_cell_5/dropout_3/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_5/split/ReadVariableOpReadVariableOp)lstm_cell_5_split_readvariableop_resource* 
_output_shapes
:
*
dtype0Ę
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0(lstm_cell_5/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0lstm_cell_5/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_5/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_5/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_5/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
lstm_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_5/split_1/ReadVariableOpReadVariableOp+lstm_cell_5_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ź
lstm_cell_5/split_1Split&lstm_cell_5/split_1/split_dim:output:0*lstm_cell_5/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell_5/BiasAddBiasAddlstm_cell_5/MatMul:product:0lstm_cell_5/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/BiasAdd_1BiasAddlstm_cell_5/MatMul_1:product:0lstm_cell_5/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/BiasAdd_2BiasAddlstm_cell_5/MatMul_2:product:0lstm_cell_5/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/BiasAdd_3BiasAddlstm_cell_5/MatMul_3:product:0lstm_cell_5/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_cell_5/mulMulzeros:output:0lstm_cell_5/dropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell_5/mul_1Mulzeros:output:0lstm_cell_5/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell_5/mul_2Mulzeros:output:0lstm_cell_5/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell_5/mul_3Mulzeros:output:0lstm_cell_5/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOpReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell_5/strided_sliceStridedSlice"lstm_cell_5/ReadVariableOp:value:0(lstm_cell_5/strided_slice/stack:output:0*lstm_cell_5/strided_slice/stack_1:output:0*lstm_cell_5/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_4MatMullstm_cell_5/mul:z:0"lstm_cell_5/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/addAddV2lstm_cell_5/BiasAdd:output:0lstm_cell_5/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell_5/SigmoidSigmoidlstm_cell_5/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOp_1ReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_5/strided_slice_1StridedSlice$lstm_cell_5/ReadVariableOp_1:value:0*lstm_cell_5/strided_slice_1/stack:output:0,lstm_cell_5/strided_slice_1/stack_1:output:0,lstm_cell_5/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_5MatMullstm_cell_5/mul_1:z:0$lstm_cell_5/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/add_1AddV2lstm_cell_5/BiasAdd_1:output:0lstm_cell_5/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_cell_5/mul_4Mullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOp_2ReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_5/strided_slice_2StridedSlice$lstm_cell_5/ReadVariableOp_2:value:0*lstm_cell_5/strided_slice_2/stack:output:0,lstm_cell_5/strided_slice_2/stack_1:output:0,lstm_cell_5/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_6MatMullstm_cell_5/mul_2:z:0$lstm_cell_5/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/add_2AddV2lstm_cell_5/BiasAdd_2:output:0lstm_cell_5/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
lstm_cell_5/TanhTanhlstm_cell_5/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
lstm_cell_5/mul_5Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_5/add_3AddV2lstm_cell_5/mul_4:z:0lstm_cell_5/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOp_3ReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_5/strided_slice_3StridedSlice$lstm_cell_5/ReadVariableOp_3:value:0*lstm_cell_5/strided_slice_3/stack:output:0,lstm_cell_5/strided_slice_3/stack_1:output:0,lstm_cell_5/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_7MatMullstm_cell_5/mul_3:z:0$lstm_cell_5/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/add_4AddV2lstm_cell_5/BiasAdd_3:output:0lstm_cell_5/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
lstm_cell_5/Tanh_1Tanhlstm_cell_5/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell_5/mul_6Mullstm_cell_5/Sigmoid_2:y:0lstm_cell_5/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ů
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_5_split_readvariableop_resource+lstm_cell_5_split_1_readvariableop_resource#lstm_cell_5_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_372806*
condR
while_cond_372805*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp^lstm_cell_5/ReadVariableOp^lstm_cell_5/ReadVariableOp_1^lstm_cell_5/ReadVariableOp_2^lstm_cell_5/ReadVariableOp_3!^lstm_cell_5/split/ReadVariableOp#^lstm_cell_5/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 28
lstm_cell_5/ReadVariableOplstm_cell_5/ReadVariableOp2<
lstm_cell_5/ReadVariableOp_1lstm_cell_5/ReadVariableOp_12<
lstm_cell_5/ReadVariableOp_2lstm_cell_5/ReadVariableOp_22<
lstm_cell_5/ReadVariableOp_3lstm_cell_5/ReadVariableOp_32D
 lstm_cell_5/split/ReadVariableOp lstm_cell_5/split/ReadVariableOp2H
"lstm_cell_5/split_1/ReadVariableOp"lstm_cell_5/split_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ň
Ţ
B__inference_lstm_4_layer_call_and_return_conditional_losses_373280

inputs<
)lstm_cell_4_split_readvariableop_resource:	9:
+lstm_cell_4_split_1_readvariableop_resource:	7
#lstm_cell_4_readvariableop_resource:

identity˘lstm_cell_4/ReadVariableOp˘lstm_cell_4/ReadVariableOp_1˘lstm_cell_4/ReadVariableOp_2˘lstm_cell_4/ReadVariableOp_3˘ lstm_cell_4/split/ReadVariableOp˘"lstm_cell_4/split_1/ReadVariableOp˘while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
shrink_axis_maskY
lstm_cell_4/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_4/dropout/MulMullstm_cell_4/ones_like:output:0"lstm_cell_4/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
lstm_cell_4/dropout/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:Ľ
0lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_4/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ë
 lstm_cell_4/dropout/GreaterEqualGreaterEqual9lstm_cell_4/dropout/random_uniform/RandomUniform:output:0+lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout/CastCast$lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout/Mul_1Mullstm_cell_4/dropout/Mul:z:0lstm_cell_4/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_4/dropout_1/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
lstm_cell_4/dropout_1/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:Š
2lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0i
$lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ń
"lstm_cell_4/dropout_1/GreaterEqualGreaterEqual;lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout_1/CastCast&lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout_1/Mul_1Mullstm_cell_4/dropout_1/Mul:z:0lstm_cell_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_4/dropout_2/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
lstm_cell_4/dropout_2/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:Š
2lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0i
$lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ń
"lstm_cell_4/dropout_2/GreaterEqualGreaterEqual;lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout_2/CastCast&lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout_2/Mul_1Mullstm_cell_4/dropout_2/Mul:z:0lstm_cell_4/dropout_2/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_4/dropout_3/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
lstm_cell_4/dropout_3/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:Š
2lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0i
$lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ń
"lstm_cell_4/dropout_3/GreaterEqualGreaterEqual;lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout_3/CastCast&lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout_3/Mul_1Mullstm_cell_4/dropout_3/Mul:z:0lstm_cell_4/dropout_3/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	9*
dtype0Ć
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_split
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ź
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/dropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
lstm_cell_4/TanhTanhlstm_cell_4/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
lstm_cell_4/Tanh_1Tanhlstm_cell_4/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ů
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_373121*
condR
while_cond_373120*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : 28
lstm_cell_4/ReadVariableOplstm_cell_4/ReadVariableOp2<
lstm_cell_4/ReadVariableOp_1lstm_cell_4/ReadVariableOp_12<
lstm_cell_4/ReadVariableOp_2lstm_cell_4/ReadVariableOp_22<
lstm_cell_4/ReadVariableOp_3lstm_cell_4/ReadVariableOp_32D
 lstm_cell_4/split/ReadVariableOp lstm_cell_4/split/ReadVariableOp2H
"lstm_cell_4/split_1/ReadVariableOp"lstm_cell_4/split_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
 
_user_specified_nameinputs
š
Ă
while_cond_373120
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_373120___redundant_placeholder04
0while_while_cond_373120___redundant_placeholder14
0while_while_cond_373120___redundant_placeholder24
0while_while_cond_373120___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:
Ý
	
while_body_376069
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_5_split_readvariableop_resource_0:
B
3while_lstm_cell_5_split_1_readvariableop_resource_0:	?
+while_lstm_cell_5_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_5_split_readvariableop_resource:
@
1while_lstm_cell_5_split_1_readvariableop_resource:	=
)while_lstm_cell_5_readvariableop_resource:
˘ while/lstm_cell_5/ReadVariableOp˘"while/lstm_cell_5/ReadVariableOp_1˘"while/lstm_cell_5/ReadVariableOp_2˘"while/lstm_cell_5/ReadVariableOp_3˘&while/lstm_cell_5/split/ReadVariableOp˘(while/lstm_cell_5/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0d
!while/lstm_cell_5/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell_5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ž
while/lstm_cell_5/ones_likeFill*while/lstm_cell_5/ones_like/Shape:output:0*while/lstm_cell_5/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
while/lstm_cell_5/dropout/MulMul$while/lstm_cell_5/ones_like:output:0(while/lstm_cell_5/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
while/lstm_cell_5/dropout/ShapeShape$while/lstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:ą
6while/lstm_cell_5/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_5/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ý
&while/lstm_cell_5/dropout/GreaterEqualGreaterEqual?while/lstm_cell_5/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/dropout/CastCast*while/lstm_cell_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
while/lstm_cell_5/dropout/Mul_1Mul!while/lstm_cell_5/dropout/Mul:z:0"while/lstm_cell_5/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell_5/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ť
while/lstm_cell_5/dropout_1/MulMul$while/lstm_cell_5/ones_like:output:0*while/lstm_cell_5/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
!while/lstm_cell_5/dropout_1/ShapeShape$while/lstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:ľ
8while/lstm_cell_5/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_5/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0o
*while/lstm_cell_5/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ă
(while/lstm_cell_5/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_5/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_5/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_5/dropout_1/CastCast,while/lstm_cell_5/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
!while/lstm_cell_5/dropout_1/Mul_1Mul#while/lstm_cell_5/dropout_1/Mul:z:0$while/lstm_cell_5/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell_5/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ť
while/lstm_cell_5/dropout_2/MulMul$while/lstm_cell_5/ones_like:output:0*while/lstm_cell_5/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
!while/lstm_cell_5/dropout_2/ShapeShape$while/lstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:ľ
8while/lstm_cell_5/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_5/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0o
*while/lstm_cell_5/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ă
(while/lstm_cell_5/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_5/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_5/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_5/dropout_2/CastCast,while/lstm_cell_5/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
!while/lstm_cell_5/dropout_2/Mul_1Mul#while/lstm_cell_5/dropout_2/Mul:z:0$while/lstm_cell_5/dropout_2/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell_5/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ť
while/lstm_cell_5/dropout_3/MulMul$while/lstm_cell_5/ones_like:output:0*while/lstm_cell_5/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
!while/lstm_cell_5/dropout_3/ShapeShape$while/lstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:ľ
8while/lstm_cell_5/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_5/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0o
*while/lstm_cell_5/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ă
(while/lstm_cell_5/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_5/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_5/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_5/dropout_3/CastCast,while/lstm_cell_5/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
!while/lstm_cell_5/dropout_3/Mul_1Mul#while/lstm_cell_5/dropout_3/Mul:z:0$while/lstm_cell_5/dropout_3/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_5/split/ReadVariableOpReadVariableOp1while_lstm_cell_5_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ü
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0.while/lstm_cell_5/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_splitŠ
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_5/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_5/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_5/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
#while/lstm_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_5/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_5_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Î
while/lstm_cell_5/split_1Split,while/lstm_cell_5/split_1/split_dim:output:00while/lstm_cell_5/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell_5/BiasAddBiasAdd"while/lstm_cell_5/MatMul:product:0"while/lstm_cell_5/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_5/BiasAdd_1BiasAdd$while/lstm_cell_5/MatMul_1:product:0"while/lstm_cell_5/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_5/BiasAdd_2BiasAdd$while/lstm_cell_5/MatMul_2:product:0"while/lstm_cell_5/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_5/BiasAdd_3BiasAdd$while/lstm_cell_5/MatMul_3:product:0"while/lstm_cell_5/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mulMulwhile_placeholder_2#while/lstm_cell_5/dropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_1Mulwhile_placeholder_2%while/lstm_cell_5/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_2Mulwhile_placeholder_2%while/lstm_cell_5/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_3Mulwhile_placeholder_2%while/lstm_cell_5/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_5/ReadVariableOpReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell_5/strided_sliceStridedSlice(while/lstm_cell_5/ReadVariableOp:value:0.while/lstm_cell_5/strided_slice/stack:output:00while/lstm_cell_5/strided_slice/stack_1:output:00while/lstm_cell_5/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_5/MatMul_4MatMulwhile/lstm_cell_5/mul:z:0(while/lstm_cell_5/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/addAddV2"while/lstm_cell_5/BiasAdd:output:0$while/lstm_cell_5/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell_5/SigmoidSigmoidwhile/lstm_cell_5/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_5/ReadVariableOp_1ReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_5/strided_slice_1StridedSlice*while/lstm_cell_5/ReadVariableOp_1:value:00while/lstm_cell_5/strided_slice_1/stack:output:02while/lstm_cell_5/strided_slice_1/stack_1:output:02while/lstm_cell_5/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_5/MatMul_5MatMulwhile/lstm_cell_5/mul_1:z:0*while/lstm_cell_5/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_1AddV2$while/lstm_cell_5/BiasAdd_1:output:0$while/lstm_cell_5/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_5/Sigmoid_1Sigmoidwhile/lstm_cell_5/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_4Mulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_5/ReadVariableOp_2ReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_5/strided_slice_2StridedSlice*while/lstm_cell_5/ReadVariableOp_2:value:00while/lstm_cell_5/strided_slice_2/stack:output:02while/lstm_cell_5/strided_slice_2/stack_1:output:02while/lstm_cell_5/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_5/MatMul_6MatMulwhile/lstm_cell_5/mul_2:z:0*while/lstm_cell_5/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_2AddV2$while/lstm_cell_5/BiasAdd_2:output:0$while/lstm_cell_5/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
while/lstm_cell_5/TanhTanhwhile/lstm_cell_5/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_5Mulwhile/lstm_cell_5/Sigmoid:y:0while/lstm_cell_5/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_3AddV2while/lstm_cell_5/mul_4:z:0while/lstm_cell_5/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_5/ReadVariableOp_3ReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_5/strided_slice_3StridedSlice*while/lstm_cell_5/ReadVariableOp_3:value:00while/lstm_cell_5/strided_slice_3/stack:output:02while/lstm_cell_5/strided_slice_3/stack_1:output:02while/lstm_cell_5/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_5/MatMul_7MatMulwhile/lstm_cell_5/mul_3:z:0*while/lstm_cell_5/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_4AddV2$while/lstm_cell_5/BiasAdd_3:output:0$while/lstm_cell_5/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_5/Sigmoid_2Sigmoidwhile/lstm_cell_5/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/lstm_cell_5/Tanh_1Tanhwhile/lstm_cell_5/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_6Mulwhile/lstm_cell_5/Sigmoid_2:y:0while/lstm_cell_5/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_6:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇy
while/Identity_4Identitywhile/lstm_cell_5/mul_6:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
while/Identity_5Identitywhile/lstm_cell_5/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˛

while/NoOpNoOp!^while/lstm_cell_5/ReadVariableOp#^while/lstm_cell_5/ReadVariableOp_1#^while/lstm_cell_5/ReadVariableOp_2#^while/lstm_cell_5/ReadVariableOp_3'^while/lstm_cell_5/split/ReadVariableOp)^while/lstm_cell_5/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_5_readvariableop_resource+while_lstm_cell_5_readvariableop_resource_0"h
1while_lstm_cell_5_split_1_readvariableop_resource3while_lstm_cell_5_split_1_readvariableop_resource_0"d
/while_lstm_cell_5_split_readvariableop_resource1while_lstm_cell_5_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2D
 while/lstm_cell_5/ReadVariableOp while/lstm_cell_5/ReadVariableOp2H
"while/lstm_cell_5/ReadVariableOp_1"while/lstm_cell_5/ReadVariableOp_12H
"while/lstm_cell_5/ReadVariableOp_2"while/lstm_cell_5/ReadVariableOp_22H
"while/lstm_cell_5/ReadVariableOp_3"while/lstm_cell_5/ReadVariableOp_32P
&while/lstm_cell_5/split/ReadVariableOp&while/lstm_cell_5/split/ReadVariableOp2T
(while/lstm_cell_5/split_1/ReadVariableOp(while/lstm_cell_5/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 
Ü
 
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_376812

inputs9
&dense_2_matmul_readvariableop_resource:	95
'dense_2_biasadd_readvariableop_resource:9
identity˘dense_2/BiasAdd/ReadVariableOp˘dense_2/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	9*
dtype0
dense_2/MatMulMatMulReshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:9*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9f
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :9
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapedense_2/Softmax:softmax:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
š
Ă
while_cond_376590
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_376590___redundant_placeholder04
0while_while_cond_376590___redundant_placeholder14
0while_while_cond_376590___redundant_placeholder24
0while_while_cond_376590___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:
¤
ľ
'__inference_lstm_4_layer_call_fn_374618

inputs
unknown:	9
	unknown_0:	
	unknown_1:

identity˘StatefulPartitionedCallň
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_373280}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
 
_user_specified_nameinputs
˛\
Š
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_371466

inputs

states
states_10
split_readvariableop_resource:	9.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘split/ReadVariableOp˘split_1/ReadVariableOpE
ones_like/ShapeShapestates*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?q
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?u
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>­
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?u
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>­
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?u
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>­
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	9*
dtype0˘
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙X
mulMulstatesdropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
mul_1Mulstatesdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
mul_2Mulstatesdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
mul_3Mulstatesdropout_3/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskf
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙X
mul_4MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mul_5MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
IdentityIdentity	mul_6:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_1Identity	mul_6:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙9:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙9
 
_user_specified_nameinputs:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namestates:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namestates
Ý
	
while_body_376591
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_5_split_readvariableop_resource_0:
B
3while_lstm_cell_5_split_1_readvariableop_resource_0:	?
+while_lstm_cell_5_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_5_split_readvariableop_resource:
@
1while_lstm_cell_5_split_1_readvariableop_resource:	=
)while_lstm_cell_5_readvariableop_resource:
˘ while/lstm_cell_5/ReadVariableOp˘"while/lstm_cell_5/ReadVariableOp_1˘"while/lstm_cell_5/ReadVariableOp_2˘"while/lstm_cell_5/ReadVariableOp_3˘&while/lstm_cell_5/split/ReadVariableOp˘(while/lstm_cell_5/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0d
!while/lstm_cell_5/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell_5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ž
while/lstm_cell_5/ones_likeFill*while/lstm_cell_5/ones_like/Shape:output:0*while/lstm_cell_5/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
while/lstm_cell_5/dropout/MulMul$while/lstm_cell_5/ones_like:output:0(while/lstm_cell_5/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
while/lstm_cell_5/dropout/ShapeShape$while/lstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:ą
6while/lstm_cell_5/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_5/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ý
&while/lstm_cell_5/dropout/GreaterEqualGreaterEqual?while/lstm_cell_5/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/dropout/CastCast*while/lstm_cell_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
while/lstm_cell_5/dropout/Mul_1Mul!while/lstm_cell_5/dropout/Mul:z:0"while/lstm_cell_5/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell_5/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ť
while/lstm_cell_5/dropout_1/MulMul$while/lstm_cell_5/ones_like:output:0*while/lstm_cell_5/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
!while/lstm_cell_5/dropout_1/ShapeShape$while/lstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:ľ
8while/lstm_cell_5/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_5/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0o
*while/lstm_cell_5/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ă
(while/lstm_cell_5/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_5/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_5/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_5/dropout_1/CastCast,while/lstm_cell_5/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
!while/lstm_cell_5/dropout_1/Mul_1Mul#while/lstm_cell_5/dropout_1/Mul:z:0$while/lstm_cell_5/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell_5/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ť
while/lstm_cell_5/dropout_2/MulMul$while/lstm_cell_5/ones_like:output:0*while/lstm_cell_5/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
!while/lstm_cell_5/dropout_2/ShapeShape$while/lstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:ľ
8while/lstm_cell_5/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_5/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0o
*while/lstm_cell_5/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ă
(while/lstm_cell_5/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_5/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_5/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_5/dropout_2/CastCast,while/lstm_cell_5/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
!while/lstm_cell_5/dropout_2/Mul_1Mul#while/lstm_cell_5/dropout_2/Mul:z:0$while/lstm_cell_5/dropout_2/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell_5/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ť
while/lstm_cell_5/dropout_3/MulMul$while/lstm_cell_5/ones_like:output:0*while/lstm_cell_5/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
!while/lstm_cell_5/dropout_3/ShapeShape$while/lstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:ľ
8while/lstm_cell_5/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_5/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0o
*while/lstm_cell_5/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ă
(while/lstm_cell_5/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_5/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_5/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_5/dropout_3/CastCast,while/lstm_cell_5/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
!while/lstm_cell_5/dropout_3/Mul_1Mul#while/lstm_cell_5/dropout_3/Mul:z:0$while/lstm_cell_5/dropout_3/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_5/split/ReadVariableOpReadVariableOp1while_lstm_cell_5_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ü
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0.while/lstm_cell_5/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_splitŠ
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_5/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_5/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_5/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
#while/lstm_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_5/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_5_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Î
while/lstm_cell_5/split_1Split,while/lstm_cell_5/split_1/split_dim:output:00while/lstm_cell_5/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell_5/BiasAddBiasAdd"while/lstm_cell_5/MatMul:product:0"while/lstm_cell_5/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_5/BiasAdd_1BiasAdd$while/lstm_cell_5/MatMul_1:product:0"while/lstm_cell_5/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_5/BiasAdd_2BiasAdd$while/lstm_cell_5/MatMul_2:product:0"while/lstm_cell_5/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_5/BiasAdd_3BiasAdd$while/lstm_cell_5/MatMul_3:product:0"while/lstm_cell_5/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mulMulwhile_placeholder_2#while/lstm_cell_5/dropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_1Mulwhile_placeholder_2%while/lstm_cell_5/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_2Mulwhile_placeholder_2%while/lstm_cell_5/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_3Mulwhile_placeholder_2%while/lstm_cell_5/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_5/ReadVariableOpReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell_5/strided_sliceStridedSlice(while/lstm_cell_5/ReadVariableOp:value:0.while/lstm_cell_5/strided_slice/stack:output:00while/lstm_cell_5/strided_slice/stack_1:output:00while/lstm_cell_5/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_5/MatMul_4MatMulwhile/lstm_cell_5/mul:z:0(while/lstm_cell_5/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/addAddV2"while/lstm_cell_5/BiasAdd:output:0$while/lstm_cell_5/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell_5/SigmoidSigmoidwhile/lstm_cell_5/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_5/ReadVariableOp_1ReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_5/strided_slice_1StridedSlice*while/lstm_cell_5/ReadVariableOp_1:value:00while/lstm_cell_5/strided_slice_1/stack:output:02while/lstm_cell_5/strided_slice_1/stack_1:output:02while/lstm_cell_5/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_5/MatMul_5MatMulwhile/lstm_cell_5/mul_1:z:0*while/lstm_cell_5/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_1AddV2$while/lstm_cell_5/BiasAdd_1:output:0$while/lstm_cell_5/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_5/Sigmoid_1Sigmoidwhile/lstm_cell_5/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_4Mulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_5/ReadVariableOp_2ReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_5/strided_slice_2StridedSlice*while/lstm_cell_5/ReadVariableOp_2:value:00while/lstm_cell_5/strided_slice_2/stack:output:02while/lstm_cell_5/strided_slice_2/stack_1:output:02while/lstm_cell_5/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_5/MatMul_6MatMulwhile/lstm_cell_5/mul_2:z:0*while/lstm_cell_5/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_2AddV2$while/lstm_cell_5/BiasAdd_2:output:0$while/lstm_cell_5/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
while/lstm_cell_5/TanhTanhwhile/lstm_cell_5/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_5Mulwhile/lstm_cell_5/Sigmoid:y:0while/lstm_cell_5/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_3AddV2while/lstm_cell_5/mul_4:z:0while/lstm_cell_5/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_5/ReadVariableOp_3ReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_5/strided_slice_3StridedSlice*while/lstm_cell_5/ReadVariableOp_3:value:00while/lstm_cell_5/strided_slice_3/stack:output:02while/lstm_cell_5/strided_slice_3/stack_1:output:02while/lstm_cell_5/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_5/MatMul_7MatMulwhile/lstm_cell_5/mul_3:z:0*while/lstm_cell_5/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_4AddV2$while/lstm_cell_5/BiasAdd_3:output:0$while/lstm_cell_5/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_5/Sigmoid_2Sigmoidwhile/lstm_cell_5/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/lstm_cell_5/Tanh_1Tanhwhile/lstm_cell_5/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_6Mulwhile/lstm_cell_5/Sigmoid_2:y:0while/lstm_cell_5/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_6:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇy
while/Identity_4Identitywhile/lstm_cell_5/mul_6:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
while/Identity_5Identitywhile/lstm_cell_5/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˛

while/NoOpNoOp!^while/lstm_cell_5/ReadVariableOp#^while/lstm_cell_5/ReadVariableOp_1#^while/lstm_cell_5/ReadVariableOp_2#^while/lstm_cell_5/ReadVariableOp_3'^while/lstm_cell_5/split/ReadVariableOp)^while/lstm_cell_5/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_5_readvariableop_resource+while_lstm_cell_5_readvariableop_resource_0"h
1while_lstm_cell_5_split_1_readvariableop_resource3while_lstm_cell_5_split_1_readvariableop_resource_0"d
/while_lstm_cell_5_split_readvariableop_resource1while_lstm_cell_5_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2D
 while/lstm_cell_5/ReadVariableOp while/lstm_cell_5/ReadVariableOp2H
"while/lstm_cell_5/ReadVariableOp_1"while/lstm_cell_5/ReadVariableOp_12H
"while/lstm_cell_5/ReadVariableOp_2"while/lstm_cell_5/ReadVariableOp_22H
"while/lstm_cell_5/ReadVariableOp_3"while/lstm_cell_5/ReadVariableOp_32P
&while/lstm_cell_5/split/ReadVariableOp&while/lstm_cell_5/split/ReadVariableOp2T
(while/lstm_cell_5/split_1/ReadVariableOp(while/lstm_cell_5/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 
š
Ă
while_cond_371992
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_371992___redundant_placeholder04
0while_while_cond_371992___redundant_placeholder14
0while_while_cond_371992___redundant_placeholder24
0while_while_cond_371992___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:
ý	
Ď
lstm_5_while_cond_373801*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3,
(lstm_5_while_less_lstm_5_strided_slice_1B
>lstm_5_while_lstm_5_while_cond_373801___redundant_placeholder0B
>lstm_5_while_lstm_5_while_cond_373801___redundant_placeholder1B
>lstm_5_while_lstm_5_while_cond_373801___redundant_placeholder2B
>lstm_5_while_lstm_5_while_cond_373801___redundant_placeholder3
lstm_5_while_identity
~
lstm_5/while/LessLesslstm_5_while_placeholder(lstm_5_while_less_lstm_5_strided_slice_1*
T0*
_output_shapes
: Y
lstm_5/while/IdentityIdentitylstm_5/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_5_while_identitylstm_5/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:
ô
ö
,__inference_lstm_cell_4_layer_call_fn_376846

inputs
states_0
states_1
unknown:	9
	unknown_0:	
	unknown_1:

identity

identity_1

identity_2˘StatefulPartitionedCallŞ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_371466p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙9:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙9
 
_user_specified_nameinputs:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states/1
Ł

ő
C__inference_dense_2_layer_call_and_return_conditional_losses_377264

inputs1
matmul_readvariableop_resource:	9-
biasadd_readvariableop_resource:9
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	9*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:9*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
8

B__inference_lstm_5_layer_call_and_return_conditional_losses_371796

inputs&
lstm_cell_5_371714:
!
lstm_cell_5_371716:	&
lstm_cell_5_371718:

identity˘#lstm_cell_5/StatefulPartitionedCall˘while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ę
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskó
#lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_5_371714lstm_cell_5_371716lstm_cell_5_371718*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_5_layer_call_and_return_conditional_losses_371713n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¸
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_5_371714lstm_cell_5_371716lstm_cell_5_371718*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_371727*
condR
while_cond_371726*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙t
NoOpNoOp$^lstm_cell_5/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 2J
#lstm_cell_5/StatefulPartitionedCall#lstm_cell_5/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
š
Ă
while_cond_372805
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_372805___redundant_placeholder04
0while_while_cond_372805___redundant_placeholder14
0while_while_cond_372805___redundant_placeholder24
0while_while_cond_372805___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:
ě"
Ţ
while_body_371525
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_4_371549_0:	9)
while_lstm_cell_4_371551_0:	.
while_lstm_cell_4_371553_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_4_371549:	9'
while_lstm_cell_4_371551:	,
while_lstm_cell_4_371553:
˘)while/lstm_cell_4/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
element_dtype0ą
)while/lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_4_371549_0while_lstm_cell_4_371551_0while_lstm_cell_4_371553_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_371466Ű
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇ
while/Identity_4Identity2while/lstm_cell_4/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/Identity_5Identity2while/lstm_cell_4/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x

while/NoOpNoOp*^while/lstm_cell_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_4_371549while_lstm_cell_4_371549_0"6
while_lstm_cell_4_371551while_lstm_cell_4_371551_0"6
while_lstm_cell_4_371553while_lstm_cell_4_371553_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2V
)while/lstm_cell_4/StatefulPartitionedCall)while/lstm_cell_4/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 
ę
Ó
%sequential_2_lstm_4_while_cond_370762D
@sequential_2_lstm_4_while_sequential_2_lstm_4_while_loop_counterJ
Fsequential_2_lstm_4_while_sequential_2_lstm_4_while_maximum_iterations)
%sequential_2_lstm_4_while_placeholder+
'sequential_2_lstm_4_while_placeholder_1+
'sequential_2_lstm_4_while_placeholder_2+
'sequential_2_lstm_4_while_placeholder_3F
Bsequential_2_lstm_4_while_less_sequential_2_lstm_4_strided_slice_1\
Xsequential_2_lstm_4_while_sequential_2_lstm_4_while_cond_370762___redundant_placeholder0\
Xsequential_2_lstm_4_while_sequential_2_lstm_4_while_cond_370762___redundant_placeholder1\
Xsequential_2_lstm_4_while_sequential_2_lstm_4_while_cond_370762___redundant_placeholder2\
Xsequential_2_lstm_4_while_sequential_2_lstm_4_while_cond_370762___redundant_placeholder3&
"sequential_2_lstm_4_while_identity
˛
sequential_2/lstm_4/while/LessLess%sequential_2_lstm_4_while_placeholderBsequential_2_lstm_4_while_less_sequential_2_lstm_4_strided_slice_1*
T0*
_output_shapes
: s
"sequential_2/lstm_4/while/IdentityIdentity"sequential_2/lstm_4/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_2_lstm_4_while_identity+sequential_2/lstm_4/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:
ä
á
B__inference_lstm_5_layer_call_and_return_conditional_losses_376228
inputs_0=
)lstm_cell_5_split_readvariableop_resource:
:
+lstm_cell_5_split_1_readvariableop_resource:	7
#lstm_cell_5_readvariableop_resource:

identity˘lstm_cell_5/ReadVariableOp˘lstm_cell_5/ReadVariableOp_1˘lstm_cell_5/ReadVariableOp_2˘lstm_cell_5/ReadVariableOp_3˘ lstm_cell_5/split/ReadVariableOp˘"lstm_cell_5/split_1/ReadVariableOp˘while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ę
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskY
lstm_cell_5/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell_5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_5/ones_likeFill$lstm_cell_5/ones_like/Shape:output:0$lstm_cell_5/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_5/dropout/MulMullstm_cell_5/ones_like:output:0"lstm_cell_5/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
lstm_cell_5/dropout/ShapeShapelstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:Ľ
0lstm_cell_5/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_5/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ë
 lstm_cell_5/dropout/GreaterEqualGreaterEqual9lstm_cell_5/dropout/random_uniform/RandomUniform:output:0+lstm_cell_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout/CastCast$lstm_cell_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout/Mul_1Mullstm_cell_5/dropout/Mul:z:0lstm_cell_5/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_5/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_5/dropout_1/MulMullstm_cell_5/ones_like:output:0$lstm_cell_5/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
lstm_cell_5/dropout_1/ShapeShapelstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:Š
2lstm_cell_5/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_5/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0i
$lstm_cell_5/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ń
"lstm_cell_5/dropout_1/GreaterEqualGreaterEqual;lstm_cell_5/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_5/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout_1/CastCast&lstm_cell_5/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout_1/Mul_1Mullstm_cell_5/dropout_1/Mul:z:0lstm_cell_5/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_5/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_5/dropout_2/MulMullstm_cell_5/ones_like:output:0$lstm_cell_5/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
lstm_cell_5/dropout_2/ShapeShapelstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:Š
2lstm_cell_5/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_5/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0i
$lstm_cell_5/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ń
"lstm_cell_5/dropout_2/GreaterEqualGreaterEqual;lstm_cell_5/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_5/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout_2/CastCast&lstm_cell_5/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout_2/Mul_1Mullstm_cell_5/dropout_2/Mul:z:0lstm_cell_5/dropout_2/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_5/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_5/dropout_3/MulMullstm_cell_5/ones_like:output:0$lstm_cell_5/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
lstm_cell_5/dropout_3/ShapeShapelstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:Š
2lstm_cell_5/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_5/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0i
$lstm_cell_5/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ń
"lstm_cell_5/dropout_3/GreaterEqualGreaterEqual;lstm_cell_5/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_5/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout_3/CastCast&lstm_cell_5/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout_3/Mul_1Mullstm_cell_5/dropout_3/Mul:z:0lstm_cell_5/dropout_3/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_5/split/ReadVariableOpReadVariableOp)lstm_cell_5_split_readvariableop_resource* 
_output_shapes
:
*
dtype0Ę
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0(lstm_cell_5/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0lstm_cell_5/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_5/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_5/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_5/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
lstm_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_5/split_1/ReadVariableOpReadVariableOp+lstm_cell_5_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ź
lstm_cell_5/split_1Split&lstm_cell_5/split_1/split_dim:output:0*lstm_cell_5/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell_5/BiasAddBiasAddlstm_cell_5/MatMul:product:0lstm_cell_5/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/BiasAdd_1BiasAddlstm_cell_5/MatMul_1:product:0lstm_cell_5/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/BiasAdd_2BiasAddlstm_cell_5/MatMul_2:product:0lstm_cell_5/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/BiasAdd_3BiasAddlstm_cell_5/MatMul_3:product:0lstm_cell_5/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_cell_5/mulMulzeros:output:0lstm_cell_5/dropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell_5/mul_1Mulzeros:output:0lstm_cell_5/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell_5/mul_2Mulzeros:output:0lstm_cell_5/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell_5/mul_3Mulzeros:output:0lstm_cell_5/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOpReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell_5/strided_sliceStridedSlice"lstm_cell_5/ReadVariableOp:value:0(lstm_cell_5/strided_slice/stack:output:0*lstm_cell_5/strided_slice/stack_1:output:0*lstm_cell_5/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_4MatMullstm_cell_5/mul:z:0"lstm_cell_5/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/addAddV2lstm_cell_5/BiasAdd:output:0lstm_cell_5/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell_5/SigmoidSigmoidlstm_cell_5/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOp_1ReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_5/strided_slice_1StridedSlice$lstm_cell_5/ReadVariableOp_1:value:0*lstm_cell_5/strided_slice_1/stack:output:0,lstm_cell_5/strided_slice_1/stack_1:output:0,lstm_cell_5/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_5MatMullstm_cell_5/mul_1:z:0$lstm_cell_5/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/add_1AddV2lstm_cell_5/BiasAdd_1:output:0lstm_cell_5/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_cell_5/mul_4Mullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOp_2ReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_5/strided_slice_2StridedSlice$lstm_cell_5/ReadVariableOp_2:value:0*lstm_cell_5/strided_slice_2/stack:output:0,lstm_cell_5/strided_slice_2/stack_1:output:0,lstm_cell_5/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_6MatMullstm_cell_5/mul_2:z:0$lstm_cell_5/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/add_2AddV2lstm_cell_5/BiasAdd_2:output:0lstm_cell_5/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
lstm_cell_5/TanhTanhlstm_cell_5/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
lstm_cell_5/mul_5Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_5/add_3AddV2lstm_cell_5/mul_4:z:0lstm_cell_5/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOp_3ReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_5/strided_slice_3StridedSlice$lstm_cell_5/ReadVariableOp_3:value:0*lstm_cell_5/strided_slice_3/stack:output:0,lstm_cell_5/strided_slice_3/stack_1:output:0,lstm_cell_5/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_7MatMullstm_cell_5/mul_3:z:0$lstm_cell_5/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/add_4AddV2lstm_cell_5/BiasAdd_3:output:0lstm_cell_5/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
lstm_cell_5/Tanh_1Tanhlstm_cell_5/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell_5/mul_6Mullstm_cell_5/Sigmoid_2:y:0lstm_cell_5/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ů
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_5_split_readvariableop_resource+lstm_cell_5_split_1_readvariableop_resource#lstm_cell_5_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_376069*
condR
while_cond_376068*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp^lstm_cell_5/ReadVariableOp^lstm_cell_5/ReadVariableOp_1^lstm_cell_5/ReadVariableOp_2^lstm_cell_5/ReadVariableOp_3!^lstm_cell_5/split/ReadVariableOp#^lstm_cell_5/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 28
lstm_cell_5/ReadVariableOplstm_cell_5/ReadVariableOp2<
lstm_cell_5/ReadVariableOp_1lstm_cell_5/ReadVariableOp_12<
lstm_cell_5/ReadVariableOp_2lstm_cell_5/ReadVariableOp_22<
lstm_cell_5/ReadVariableOp_3lstm_cell_5/ReadVariableOp_32D
 lstm_cell_5/split/ReadVariableOp lstm_cell_5/split/ReadVariableOp2H
"lstm_cell_5/split_1/ReadVariableOp"lstm_cell_5/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0
ş?
Ź
G__inference_lstm_cell_5_layer_call_and_return_conditional_losses_377137

inputs
states_0
states_11
split_readvariableop_resource:
.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘split/ReadVariableOp˘split_1/ReadVariableOpG
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
*
dtype0Ś
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
mulMulstates_0ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
mul_1Mulstates_0ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
mul_2Mulstates_0ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
mul_3Mulstates_0ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskf
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙X
mul_4MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mul_5MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
IdentityIdentity	mul_6:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_1Identity	mul_6:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states/1
š
Ă
while_cond_374719
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_374719___redundant_placeholder04
0while_while_cond_374719___redundant_placeholder14
0while_while_cond_374719___redundant_placeholder24
0while_while_cond_374719___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:


Ę
-__inference_sequential_2_layer_call_fn_373475

inputs
unknown:	9
	unknown_0:	
	unknown_1:

	unknown_2:

	unknown_3:	
	unknown_4:

	unknown_5:	9
	unknown_6:9
identity˘StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_373337|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
 
_user_specified_nameinputs
Ę\
Ź
G__inference_lstm_cell_5_layer_call_and_return_conditional_losses_377244

inputs
states_0
states_11
split_readvariableop_resource:
.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘split/ReadVariableOp˘split_1/ReadVariableOpG
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?q
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?u
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>­
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?u
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>­
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?u
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>­
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
*
dtype0Ś
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mulMulstates_0dropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
mul_1Mulstates_0dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
mul_2Mulstates_0dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
mul_3Mulstates_0dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskf
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙X
mul_4MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mul_5MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
IdentityIdentity	mul_6:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_1Identity	mul_6:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states/1
ńl
	
while_body_375242
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_4_split_readvariableop_resource_0:	9B
3while_lstm_cell_4_split_1_readvariableop_resource_0:	?
+while_lstm_cell_4_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_4_split_readvariableop_resource:	9@
1while_lstm_cell_4_split_1_readvariableop_resource:	=
)while_lstm_cell_4_readvariableop_resource:
˘ while/lstm_cell_4/ReadVariableOp˘"while/lstm_cell_4/ReadVariableOp_1˘"while/lstm_cell_4/ReadVariableOp_2˘"while/lstm_cell_4/ReadVariableOp_3˘&while/lstm_cell_4/split/ReadVariableOp˘(while/lstm_cell_4/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
element_dtype0d
!while/lstm_cell_4/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ž
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	9*
dtype0Ř
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_splitŠ
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Î
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mulMulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_1Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_2Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_3Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
while/lstm_cell_4/TanhTanhwhile/lstm_cell_4/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0while/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/lstm_cell_4/Tanh_1Tanhwhile/lstm_cell_4/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0while/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇy
while/Identity_4Identitywhile/lstm_cell_4/mul_6:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˛

while/NoOpNoOp!^while/lstm_cell_4/ReadVariableOp#^while/lstm_cell_4/ReadVariableOp_1#^while/lstm_cell_4/ReadVariableOp_2#^while/lstm_cell_4/ReadVariableOp_3'^while/lstm_cell_4/split/ReadVariableOp)^while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2D
 while/lstm_cell_4/ReadVariableOp while/lstm_cell_4/ReadVariableOp2H
"while/lstm_cell_4/ReadVariableOp_1"while/lstm_cell_4/ReadVariableOp_12H
"while/lstm_cell_4/ReadVariableOp_2"while/lstm_cell_4/ReadVariableOp_22H
"while/lstm_cell_4/ReadVariableOp_3"while/lstm_cell_4/ReadVariableOp_32P
&while/lstm_cell_4/split/ReadVariableOp&while/lstm_cell_4/split/ReadVariableOp2T
(while/lstm_cell_4/split_1/ReadVariableOp(while/lstm_cell_4/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 
ůl
	
while_body_376330
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_5_split_readvariableop_resource_0:
B
3while_lstm_cell_5_split_1_readvariableop_resource_0:	?
+while_lstm_cell_5_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_5_split_readvariableop_resource:
@
1while_lstm_cell_5_split_1_readvariableop_resource:	=
)while_lstm_cell_5_readvariableop_resource:
˘ while/lstm_cell_5/ReadVariableOp˘"while/lstm_cell_5/ReadVariableOp_1˘"while/lstm_cell_5/ReadVariableOp_2˘"while/lstm_cell_5/ReadVariableOp_3˘&while/lstm_cell_5/split/ReadVariableOp˘(while/lstm_cell_5/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0d
!while/lstm_cell_5/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell_5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ž
while/lstm_cell_5/ones_likeFill*while/lstm_cell_5/ones_like/Shape:output:0*while/lstm_cell_5/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_5/split/ReadVariableOpReadVariableOp1while_lstm_cell_5_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ü
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0.while/lstm_cell_5/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_splitŠ
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_5/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_5/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_5/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
#while/lstm_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_5/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_5_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Î
while/lstm_cell_5/split_1Split,while/lstm_cell_5/split_1/split_dim:output:00while/lstm_cell_5/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell_5/BiasAddBiasAdd"while/lstm_cell_5/MatMul:product:0"while/lstm_cell_5/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_5/BiasAdd_1BiasAdd$while/lstm_cell_5/MatMul_1:product:0"while/lstm_cell_5/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_5/BiasAdd_2BiasAdd$while/lstm_cell_5/MatMul_2:product:0"while/lstm_cell_5/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_5/BiasAdd_3BiasAdd$while/lstm_cell_5/MatMul_3:product:0"while/lstm_cell_5/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mulMulwhile_placeholder_2$while/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_1Mulwhile_placeholder_2$while/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_2Mulwhile_placeholder_2$while/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_3Mulwhile_placeholder_2$while/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_5/ReadVariableOpReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell_5/strided_sliceStridedSlice(while/lstm_cell_5/ReadVariableOp:value:0.while/lstm_cell_5/strided_slice/stack:output:00while/lstm_cell_5/strided_slice/stack_1:output:00while/lstm_cell_5/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_5/MatMul_4MatMulwhile/lstm_cell_5/mul:z:0(while/lstm_cell_5/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/addAddV2"while/lstm_cell_5/BiasAdd:output:0$while/lstm_cell_5/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell_5/SigmoidSigmoidwhile/lstm_cell_5/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_5/ReadVariableOp_1ReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_5/strided_slice_1StridedSlice*while/lstm_cell_5/ReadVariableOp_1:value:00while/lstm_cell_5/strided_slice_1/stack:output:02while/lstm_cell_5/strided_slice_1/stack_1:output:02while/lstm_cell_5/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_5/MatMul_5MatMulwhile/lstm_cell_5/mul_1:z:0*while/lstm_cell_5/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_1AddV2$while/lstm_cell_5/BiasAdd_1:output:0$while/lstm_cell_5/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_5/Sigmoid_1Sigmoidwhile/lstm_cell_5/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_4Mulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_5/ReadVariableOp_2ReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_5/strided_slice_2StridedSlice*while/lstm_cell_5/ReadVariableOp_2:value:00while/lstm_cell_5/strided_slice_2/stack:output:02while/lstm_cell_5/strided_slice_2/stack_1:output:02while/lstm_cell_5/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_5/MatMul_6MatMulwhile/lstm_cell_5/mul_2:z:0*while/lstm_cell_5/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_2AddV2$while/lstm_cell_5/BiasAdd_2:output:0$while/lstm_cell_5/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
while/lstm_cell_5/TanhTanhwhile/lstm_cell_5/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_5Mulwhile/lstm_cell_5/Sigmoid:y:0while/lstm_cell_5/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_3AddV2while/lstm_cell_5/mul_4:z:0while/lstm_cell_5/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_5/ReadVariableOp_3ReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_5/strided_slice_3StridedSlice*while/lstm_cell_5/ReadVariableOp_3:value:00while/lstm_cell_5/strided_slice_3/stack:output:02while/lstm_cell_5/strided_slice_3/stack_1:output:02while/lstm_cell_5/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_5/MatMul_7MatMulwhile/lstm_cell_5/mul_3:z:0*while/lstm_cell_5/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_4AddV2$while/lstm_cell_5/BiasAdd_3:output:0$while/lstm_cell_5/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_5/Sigmoid_2Sigmoidwhile/lstm_cell_5/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/lstm_cell_5/Tanh_1Tanhwhile/lstm_cell_5/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_6Mulwhile/lstm_cell_5/Sigmoid_2:y:0while/lstm_cell_5/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_6:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇy
while/Identity_4Identitywhile/lstm_cell_5/mul_6:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
while/Identity_5Identitywhile/lstm_cell_5/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˛

while/NoOpNoOp!^while/lstm_cell_5/ReadVariableOp#^while/lstm_cell_5/ReadVariableOp_1#^while/lstm_cell_5/ReadVariableOp_2#^while/lstm_cell_5/ReadVariableOp_3'^while/lstm_cell_5/split/ReadVariableOp)^while/lstm_cell_5/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_5_readvariableop_resource+while_lstm_cell_5_readvariableop_resource_0"h
1while_lstm_cell_5_split_1_readvariableop_resource3while_lstm_cell_5_split_1_readvariableop_resource_0"d
/while_lstm_cell_5_split_readvariableop_resource1while_lstm_cell_5_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2D
 while/lstm_cell_5/ReadVariableOp while/lstm_cell_5/ReadVariableOp2H
"while/lstm_cell_5/ReadVariableOp_1"while/lstm_cell_5/ReadVariableOp_12H
"while/lstm_cell_5/ReadVariableOp_2"while/lstm_cell_5/ReadVariableOp_22H
"while/lstm_cell_5/ReadVariableOp_3"while/lstm_cell_5/ReadVariableOp_32P
&while/lstm_cell_5/split/ReadVariableOp&while/lstm_cell_5/split/ReadVariableOp2T
(while/lstm_cell_5/split_1/ReadVariableOp(while/lstm_cell_5/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 


Đ
-__inference_sequential_2_layer_call_fn_372659
lstm_4_input
unknown:	9
	unknown_0:	
	unknown_1:

	unknown_2:

	unknown_3:	
	unknown_4:

	unknown_5:	9
	unknown_6:9
identity˘StatefulPartitionedCallž
StatefulPartitionedCallStatefulPartitionedCalllstm_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_372640|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
&
_user_specified_namelstm_4_input
Ş
Đ
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_372107

inputs!
dense_2_372097:	9
dense_2_372099:9
identity˘dense_2/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ö
dense_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_2_372097dense_2_372099*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_372096\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :9
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape(dense_2/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9h
NoOpNoOp ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ş
ˇ
'__inference_lstm_4_layer_call_fn_374596
inputs_0
unknown:	9
	unknown_0:	
	unknown_1:

identity˘StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_371594}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
"
_user_specified_name
inputs/0
ž
Ř
%sequential_2_lstm_5_while_body_370988D
@sequential_2_lstm_5_while_sequential_2_lstm_5_while_loop_counterJ
Fsequential_2_lstm_5_while_sequential_2_lstm_5_while_maximum_iterations)
%sequential_2_lstm_5_while_placeholder+
'sequential_2_lstm_5_while_placeholder_1+
'sequential_2_lstm_5_while_placeholder_2+
'sequential_2_lstm_5_while_placeholder_3C
?sequential_2_lstm_5_while_sequential_2_lstm_5_strided_slice_1_0
{sequential_2_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_5_tensorarrayunstack_tensorlistfromtensor_0Y
Esequential_2_lstm_5_while_lstm_cell_5_split_readvariableop_resource_0:
V
Gsequential_2_lstm_5_while_lstm_cell_5_split_1_readvariableop_resource_0:	S
?sequential_2_lstm_5_while_lstm_cell_5_readvariableop_resource_0:
&
"sequential_2_lstm_5_while_identity(
$sequential_2_lstm_5_while_identity_1(
$sequential_2_lstm_5_while_identity_2(
$sequential_2_lstm_5_while_identity_3(
$sequential_2_lstm_5_while_identity_4(
$sequential_2_lstm_5_while_identity_5A
=sequential_2_lstm_5_while_sequential_2_lstm_5_strided_slice_1}
ysequential_2_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_5_tensorarrayunstack_tensorlistfromtensorW
Csequential_2_lstm_5_while_lstm_cell_5_split_readvariableop_resource:
T
Esequential_2_lstm_5_while_lstm_cell_5_split_1_readvariableop_resource:	Q
=sequential_2_lstm_5_while_lstm_cell_5_readvariableop_resource:
˘4sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp˘6sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp_1˘6sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp_2˘6sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp_3˘:sequential_2/lstm_5/while/lstm_cell_5/split/ReadVariableOp˘<sequential_2/lstm_5/while/lstm_cell_5/split_1/ReadVariableOp
Ksequential_2/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
=sequential_2/lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_2_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_5_tensorarrayunstack_tensorlistfromtensor_0%sequential_2_lstm_5_while_placeholderTsequential_2/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0
5sequential_2/lstm_5/while/lstm_cell_5/ones_like/ShapeShape'sequential_2_lstm_5_while_placeholder_2*
T0*
_output_shapes
:z
5sequential_2/lstm_5/while/lstm_cell_5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ę
/sequential_2/lstm_5/while/lstm_cell_5/ones_likeFill>sequential_2/lstm_5/while/lstm_cell_5/ones_like/Shape:output:0>sequential_2/lstm_5/while/lstm_cell_5/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
5sequential_2/lstm_5/while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Â
:sequential_2/lstm_5/while/lstm_cell_5/split/ReadVariableOpReadVariableOpEsequential_2_lstm_5_while_lstm_cell_5_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
+sequential_2/lstm_5/while/lstm_cell_5/splitSplit>sequential_2/lstm_5/while/lstm_cell_5/split/split_dim:output:0Bsequential_2/lstm_5/while/lstm_cell_5/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_splitĺ
,sequential_2/lstm_5/while/lstm_cell_5/MatMulMatMulDsequential_2/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_2/lstm_5/while/lstm_cell_5/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
.sequential_2/lstm_5/while/lstm_cell_5/MatMul_1MatMulDsequential_2/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_2/lstm_5/while/lstm_cell_5/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
.sequential_2/lstm_5/while/lstm_cell_5/MatMul_2MatMulDsequential_2/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_2/lstm_5/while/lstm_cell_5/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
.sequential_2/lstm_5/while/lstm_cell_5/MatMul_3MatMulDsequential_2/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_2/lstm_5/while/lstm_cell_5/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
7sequential_2/lstm_5/while/lstm_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Á
<sequential_2/lstm_5/while/lstm_cell_5/split_1/ReadVariableOpReadVariableOpGsequential_2_lstm_5_while_lstm_cell_5_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0
-sequential_2/lstm_5/while/lstm_cell_5/split_1Split@sequential_2/lstm_5/while/lstm_cell_5/split_1/split_dim:output:0Dsequential_2/lstm_5/while/lstm_cell_5/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_splitŰ
-sequential_2/lstm_5/while/lstm_cell_5/BiasAddBiasAdd6sequential_2/lstm_5/while/lstm_cell_5/MatMul:product:06sequential_2/lstm_5/while/lstm_cell_5/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ß
/sequential_2/lstm_5/while/lstm_cell_5/BiasAdd_1BiasAdd8sequential_2/lstm_5/while/lstm_cell_5/MatMul_1:product:06sequential_2/lstm_5/while/lstm_cell_5/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ß
/sequential_2/lstm_5/while/lstm_cell_5/BiasAdd_2BiasAdd8sequential_2/lstm_5/while/lstm_cell_5/MatMul_2:product:06sequential_2/lstm_5/while/lstm_cell_5/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ß
/sequential_2/lstm_5/while/lstm_cell_5/BiasAdd_3BiasAdd8sequential_2/lstm_5/while/lstm_cell_5/MatMul_3:product:06sequential_2/lstm_5/while/lstm_cell_5/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ć
)sequential_2/lstm_5/while/lstm_cell_5/mulMul'sequential_2_lstm_5_while_placeholder_28sequential_2/lstm_5/while/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Č
+sequential_2/lstm_5/while/lstm_cell_5/mul_1Mul'sequential_2_lstm_5_while_placeholder_28sequential_2/lstm_5/while/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Č
+sequential_2/lstm_5/while/lstm_cell_5/mul_2Mul'sequential_2_lstm_5_while_placeholder_28sequential_2/lstm_5/while/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Č
+sequential_2/lstm_5/while/lstm_cell_5/mul_3Mul'sequential_2_lstm_5_while_placeholder_28sequential_2/lstm_5/while/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ś
4sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOpReadVariableOp?sequential_2_lstm_5_while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
9sequential_2/lstm_5/while/lstm_cell_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
;sequential_2/lstm_5/while/lstm_cell_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
;sequential_2/lstm_5/while/lstm_cell_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ť
3sequential_2/lstm_5/while/lstm_cell_5/strided_sliceStridedSlice<sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp:value:0Bsequential_2/lstm_5/while/lstm_cell_5/strided_slice/stack:output:0Dsequential_2/lstm_5/while/lstm_cell_5/strided_slice/stack_1:output:0Dsequential_2/lstm_5/while/lstm_cell_5/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŘ
.sequential_2/lstm_5/while/lstm_cell_5/MatMul_4MatMul-sequential_2/lstm_5/while/lstm_cell_5/mul:z:0<sequential_2/lstm_5/while/lstm_cell_5/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙×
)sequential_2/lstm_5/while/lstm_cell_5/addAddV26sequential_2/lstm_5/while/lstm_cell_5/BiasAdd:output:08sequential_2/lstm_5/while/lstm_cell_5/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
-sequential_2/lstm_5/while/lstm_cell_5/SigmoidSigmoid-sequential_2/lstm_5/while/lstm_cell_5/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
6sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp_1ReadVariableOp?sequential_2_lstm_5_while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
;sequential_2/lstm_5/while/lstm_cell_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
=sequential_2/lstm_5/while/lstm_cell_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
=sequential_2/lstm_5/while/lstm_cell_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ľ
5sequential_2/lstm_5/while/lstm_cell_5/strided_slice_1StridedSlice>sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp_1:value:0Dsequential_2/lstm_5/while/lstm_cell_5/strided_slice_1/stack:output:0Fsequential_2/lstm_5/while/lstm_cell_5/strided_slice_1/stack_1:output:0Fsequential_2/lstm_5/while/lstm_cell_5/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÜ
.sequential_2/lstm_5/while/lstm_cell_5/MatMul_5MatMul/sequential_2/lstm_5/while/lstm_cell_5/mul_1:z:0>sequential_2/lstm_5/while/lstm_cell_5/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ű
+sequential_2/lstm_5/while/lstm_cell_5/add_1AddV28sequential_2/lstm_5/while/lstm_cell_5/BiasAdd_1:output:08sequential_2/lstm_5/while/lstm_cell_5/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
/sequential_2/lstm_5/while/lstm_cell_5/Sigmoid_1Sigmoid/sequential_2/lstm_5/while/lstm_cell_5/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
+sequential_2/lstm_5/while/lstm_cell_5/mul_4Mul3sequential_2/lstm_5/while/lstm_cell_5/Sigmoid_1:y:0'sequential_2_lstm_5_while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
6sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp_2ReadVariableOp?sequential_2_lstm_5_while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
;sequential_2/lstm_5/while/lstm_cell_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
=sequential_2/lstm_5/while/lstm_cell_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
=sequential_2/lstm_5/while/lstm_cell_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ľ
5sequential_2/lstm_5/while/lstm_cell_5/strided_slice_2StridedSlice>sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp_2:value:0Dsequential_2/lstm_5/while/lstm_cell_5/strided_slice_2/stack:output:0Fsequential_2/lstm_5/while/lstm_cell_5/strided_slice_2/stack_1:output:0Fsequential_2/lstm_5/while/lstm_cell_5/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÜ
.sequential_2/lstm_5/while/lstm_cell_5/MatMul_6MatMul/sequential_2/lstm_5/while/lstm_cell_5/mul_2:z:0>sequential_2/lstm_5/while/lstm_cell_5/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ű
+sequential_2/lstm_5/while/lstm_cell_5/add_2AddV28sequential_2/lstm_5/while/lstm_cell_5/BiasAdd_2:output:08sequential_2/lstm_5/while/lstm_cell_5/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
*sequential_2/lstm_5/while/lstm_cell_5/TanhTanh/sequential_2/lstm_5/while/lstm_cell_5/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Č
+sequential_2/lstm_5/while/lstm_cell_5/mul_5Mul1sequential_2/lstm_5/while/lstm_cell_5/Sigmoid:y:0.sequential_2/lstm_5/while/lstm_cell_5/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙É
+sequential_2/lstm_5/while/lstm_cell_5/add_3AddV2/sequential_2/lstm_5/while/lstm_cell_5/mul_4:z:0/sequential_2/lstm_5/while/lstm_cell_5/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
6sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp_3ReadVariableOp?sequential_2_lstm_5_while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
;sequential_2/lstm_5/while/lstm_cell_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
=sequential_2/lstm_5/while/lstm_cell_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
=sequential_2/lstm_5/while/lstm_cell_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ľ
5sequential_2/lstm_5/while/lstm_cell_5/strided_slice_3StridedSlice>sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp_3:value:0Dsequential_2/lstm_5/while/lstm_cell_5/strided_slice_3/stack:output:0Fsequential_2/lstm_5/while/lstm_cell_5/strided_slice_3/stack_1:output:0Fsequential_2/lstm_5/while/lstm_cell_5/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÜ
.sequential_2/lstm_5/while/lstm_cell_5/MatMul_7MatMul/sequential_2/lstm_5/while/lstm_cell_5/mul_3:z:0>sequential_2/lstm_5/while/lstm_cell_5/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ű
+sequential_2/lstm_5/while/lstm_cell_5/add_4AddV28sequential_2/lstm_5/while/lstm_cell_5/BiasAdd_3:output:08sequential_2/lstm_5/while/lstm_cell_5/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
/sequential_2/lstm_5/while/lstm_cell_5/Sigmoid_2Sigmoid/sequential_2/lstm_5/while/lstm_cell_5/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
,sequential_2/lstm_5/while/lstm_cell_5/Tanh_1Tanh/sequential_2/lstm_5/while/lstm_cell_5/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ě
+sequential_2/lstm_5/while/lstm_cell_5/mul_6Mul3sequential_2/lstm_5/while/lstm_cell_5/Sigmoid_2:y:00sequential_2/lstm_5/while/lstm_cell_5/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
>sequential_2/lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_2_lstm_5_while_placeholder_1%sequential_2_lstm_5_while_placeholder/sequential_2/lstm_5/while/lstm_cell_5/mul_6:z:0*
_output_shapes
: *
element_dtype0:éčŇa
sequential_2/lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_2/lstm_5/while/addAddV2%sequential_2_lstm_5_while_placeholder(sequential_2/lstm_5/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_2/lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ˇ
sequential_2/lstm_5/while/add_1AddV2@sequential_2_lstm_5_while_sequential_2_lstm_5_while_loop_counter*sequential_2/lstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: 
"sequential_2/lstm_5/while/IdentityIdentity#sequential_2/lstm_5/while/add_1:z:0^sequential_2/lstm_5/while/NoOp*
T0*
_output_shapes
: ş
$sequential_2/lstm_5/while/Identity_1IdentityFsequential_2_lstm_5_while_sequential_2_lstm_5_while_maximum_iterations^sequential_2/lstm_5/while/NoOp*
T0*
_output_shapes
: 
$sequential_2/lstm_5/while/Identity_2Identity!sequential_2/lstm_5/while/add:z:0^sequential_2/lstm_5/while/NoOp*
T0*
_output_shapes
: Ő
$sequential_2/lstm_5/while/Identity_3IdentityNsequential_2/lstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_2/lstm_5/while/NoOp*
T0*
_output_shapes
: :éčŇľ
$sequential_2/lstm_5/while/Identity_4Identity/sequential_2/lstm_5/while/lstm_cell_5/mul_6:z:0^sequential_2/lstm_5/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ
$sequential_2/lstm_5/while/Identity_5Identity/sequential_2/lstm_5/while/lstm_cell_5/add_3:z:0^sequential_2/lstm_5/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ž
sequential_2/lstm_5/while/NoOpNoOp5^sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp7^sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp_17^sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp_27^sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp_3;^sequential_2/lstm_5/while/lstm_cell_5/split/ReadVariableOp=^sequential_2/lstm_5/while/lstm_cell_5/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Q
"sequential_2_lstm_5_while_identity+sequential_2/lstm_5/while/Identity:output:0"U
$sequential_2_lstm_5_while_identity_1-sequential_2/lstm_5/while/Identity_1:output:0"U
$sequential_2_lstm_5_while_identity_2-sequential_2/lstm_5/while/Identity_2:output:0"U
$sequential_2_lstm_5_while_identity_3-sequential_2/lstm_5/while/Identity_3:output:0"U
$sequential_2_lstm_5_while_identity_4-sequential_2/lstm_5/while/Identity_4:output:0"U
$sequential_2_lstm_5_while_identity_5-sequential_2/lstm_5/while/Identity_5:output:0"
=sequential_2_lstm_5_while_lstm_cell_5_readvariableop_resource?sequential_2_lstm_5_while_lstm_cell_5_readvariableop_resource_0"
Esequential_2_lstm_5_while_lstm_cell_5_split_1_readvariableop_resourceGsequential_2_lstm_5_while_lstm_cell_5_split_1_readvariableop_resource_0"
Csequential_2_lstm_5_while_lstm_cell_5_split_readvariableop_resourceEsequential_2_lstm_5_while_lstm_cell_5_split_readvariableop_resource_0"
=sequential_2_lstm_5_while_sequential_2_lstm_5_strided_slice_1?sequential_2_lstm_5_while_sequential_2_lstm_5_strided_slice_1_0"ř
ysequential_2_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_5_tensorarrayunstack_tensorlistfromtensor{sequential_2_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2l
4sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp4sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp2p
6sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp_16sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp_12p
6sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp_26sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp_22p
6sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp_36sequential_2/lstm_5/while/lstm_cell_5/ReadVariableOp_32x
:sequential_2/lstm_5/while/lstm_cell_5/split/ReadVariableOp:sequential_2/lstm_5/while/lstm_cell_5/split/ReadVariableOp2|
<sequential_2/lstm_5/while/lstm_cell_5/split_1/ReadVariableOp<sequential_2/lstm_5/while/lstm_cell_5/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 


Ę
-__inference_sequential_2_layer_call_fn_373454

inputs
unknown:	9
	unknown_0:	
	unknown_1:

	unknown_2:

	unknown_3:	
	unknown_4:

	unknown_5:	9
	unknown_6:9
identity˘StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_372640|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
 
_user_specified_nameinputs
ď"
ŕ
while_body_371727
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_5_371751_0:
)
while_lstm_cell_5_371753_0:	.
while_lstm_cell_5_371755_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_5_371751:
'
while_lstm_cell_5_371753:	,
while_lstm_cell_5_371755:
˘)while/lstm_cell_5/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0ą
)while/lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_5_371751_0while_lstm_cell_5_371753_0while_lstm_cell_5_371755_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_5_layer_call_and_return_conditional_losses_371713Ű
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_5/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇ
while/Identity_4Identity2while/lstm_cell_5/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/Identity_5Identity2while/lstm_cell_5/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x

while/NoOpNoOp*^while/lstm_cell_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_5_371751while_lstm_cell_5_371751_0"6
while_lstm_cell_5_371753while_lstm_cell_5_371753_0"6
while_lstm_cell_5_371755while_lstm_cell_5_371755_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2V
)while/lstm_cell_5/StatefulPartitionedCall)while/lstm_cell_5/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 
z
Ţ
B__inference_lstm_4_layer_call_and_return_conditional_losses_375369

inputs<
)lstm_cell_4_split_readvariableop_resource:	9:
+lstm_cell_4_split_1_readvariableop_resource:	7
#lstm_cell_4_readvariableop_resource:

identity˘lstm_cell_4/ReadVariableOp˘lstm_cell_4/ReadVariableOp_1˘lstm_cell_4/ReadVariableOp_2˘lstm_cell_4/ReadVariableOp_3˘ lstm_cell_4/split/ReadVariableOp˘"lstm_cell_4/split_1/ReadVariableOp˘while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
shrink_axis_maskY
lstm_cell_4/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	9*
dtype0Ć
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_split
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ź
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
lstm_cell_4/TanhTanhlstm_cell_4/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
lstm_cell_4/Tanh_1Tanhlstm_cell_4/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ů
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_375242*
condR
while_cond_375241*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : 28
lstm_cell_4/ReadVariableOplstm_cell_4/ReadVariableOp2<
lstm_cell_4/ReadVariableOp_1lstm_cell_4/ReadVariableOp_12<
lstm_cell_4/ReadVariableOp_2lstm_cell_4/ReadVariableOp_22<
lstm_cell_4/ReadVariableOp_3lstm_cell_4/ReadVariableOp_32D
 lstm_cell_4/split/ReadVariableOp lstm_cell_4/split/ReadVariableOp2H
"lstm_cell_4/split_1/ReadVariableOp"lstm_cell_4/split_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
 
_user_specified_nameinputs
Ő
	
while_body_375503
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_4_split_readvariableop_resource_0:	9B
3while_lstm_cell_4_split_1_readvariableop_resource_0:	?
+while_lstm_cell_4_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_4_split_readvariableop_resource:	9@
1while_lstm_cell_4_split_1_readvariableop_resource:	=
)while_lstm_cell_4_readvariableop_resource:
˘ while/lstm_cell_4/ReadVariableOp˘"while/lstm_cell_4/ReadVariableOp_1˘"while/lstm_cell_4/ReadVariableOp_2˘"while/lstm_cell_4/ReadVariableOp_3˘&while/lstm_cell_4/split/ReadVariableOp˘(while/lstm_cell_4/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
element_dtype0d
!while/lstm_cell_4/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ž
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
while/lstm_cell_4/dropout/MulMul$while/lstm_cell_4/ones_like:output:0(while/lstm_cell_4/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
while/lstm_cell_4/dropout/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:ą
6while/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_4/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ý
&while/lstm_cell_4/dropout/GreaterEqualGreaterEqual?while/lstm_cell_4/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/dropout/CastCast*while/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
while/lstm_cell_4/dropout/Mul_1Mul!while/lstm_cell_4/dropout/Mul:z:0"while/lstm_cell_4/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ť
while/lstm_cell_4/dropout_1/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
!while/lstm_cell_4/dropout_1/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:ľ
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0o
*while/lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ă
(while/lstm_cell_4/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_4/dropout_1/CastCast,while/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
!while/lstm_cell_4/dropout_1/Mul_1Mul#while/lstm_cell_4/dropout_1/Mul:z:0$while/lstm_cell_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ť
while/lstm_cell_4/dropout_2/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
!while/lstm_cell_4/dropout_2/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:ľ
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0o
*while/lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ă
(while/lstm_cell_4/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_4/dropout_2/CastCast,while/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
!while/lstm_cell_4/dropout_2/Mul_1Mul#while/lstm_cell_4/dropout_2/Mul:z:0$while/lstm_cell_4/dropout_2/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ť
while/lstm_cell_4/dropout_3/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
!while/lstm_cell_4/dropout_3/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:ľ
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0o
*while/lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ă
(while/lstm_cell_4/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_4/dropout_3/CastCast,while/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
!while/lstm_cell_4/dropout_3/Mul_1Mul#while/lstm_cell_4/dropout_3/Mul:z:0$while/lstm_cell_4/dropout_3/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	9*
dtype0Ř
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_splitŠ
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Î
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mulMulwhile_placeholder_2#while/lstm_cell_4/dropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_1Mulwhile_placeholder_2%while/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_2Mulwhile_placeholder_2%while/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_3Mulwhile_placeholder_2%while/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
while/lstm_cell_4/TanhTanhwhile/lstm_cell_4/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0while/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/lstm_cell_4/Tanh_1Tanhwhile/lstm_cell_4/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0while/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇy
while/Identity_4Identitywhile/lstm_cell_4/mul_6:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˛

while/NoOpNoOp!^while/lstm_cell_4/ReadVariableOp#^while/lstm_cell_4/ReadVariableOp_1#^while/lstm_cell_4/ReadVariableOp_2#^while/lstm_cell_4/ReadVariableOp_3'^while/lstm_cell_4/split/ReadVariableOp)^while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2D
 while/lstm_cell_4/ReadVariableOp while/lstm_cell_4/ReadVariableOp2H
"while/lstm_cell_4/ReadVariableOp_1"while/lstm_cell_4/ReadVariableOp_12H
"while/lstm_cell_4/ReadVariableOp_2"while/lstm_cell_4/ReadVariableOp_22H
"while/lstm_cell_4/ReadVariableOp_3"while/lstm_cell_4/ReadVariableOp_32P
&while/lstm_cell_4/split/ReadVariableOp&while/lstm_cell_4/split/ReadVariableOp2T
(while/lstm_cell_4/split_1/ReadVariableOp(while/lstm_cell_4/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 
z
ŕ
B__inference_lstm_4_layer_call_and_return_conditional_losses_374847
inputs_0<
)lstm_cell_4_split_readvariableop_resource:	9:
+lstm_cell_4_split_1_readvariableop_resource:	7
#lstm_cell_4_readvariableop_resource:

identity˘lstm_cell_4/ReadVariableOp˘lstm_cell_4/ReadVariableOp_1˘lstm_cell_4/ReadVariableOp_2˘lstm_cell_4/ReadVariableOp_3˘ lstm_cell_4/split/ReadVariableOp˘"lstm_cell_4/split_1/ReadVariableOp˘while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
shrink_axis_maskY
lstm_cell_4/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	9*
dtype0Ć
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_split
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ź
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
lstm_cell_4/TanhTanhlstm_cell_4/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
lstm_cell_4/Tanh_1Tanhlstm_cell_4/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ů
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_374720*
condR
while_cond_374719*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : 28
lstm_cell_4/ReadVariableOplstm_cell_4/ReadVariableOp2<
lstm_cell_4/ReadVariableOp_1lstm_cell_4/ReadVariableOp_12<
lstm_cell_4/ReadVariableOp_2lstm_cell_4/ReadVariableOp_22<
lstm_cell_4/ReadVariableOp_3lstm_cell_4/ReadVariableOp_32D
 lstm_cell_4/split/ReadVariableOp lstm_cell_4/split/ReadVariableOp2H
"lstm_cell_4/split_1/ReadVariableOp"lstm_cell_4/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
"
_user_specified_name
inputs/0
ý	
Ď
lstm_4_while_cond_374082*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3,
(lstm_4_while_less_lstm_4_strided_slice_1B
>lstm_4_while_lstm_4_while_cond_374082___redundant_placeholder0B
>lstm_4_while_lstm_4_while_cond_374082___redundant_placeholder1B
>lstm_4_while_lstm_4_while_cond_374082___redundant_placeholder2B
>lstm_4_while_lstm_4_while_cond_374082___redundant_placeholder3
lstm_4_while_identity
~
lstm_4/while/LessLesslstm_4_while_placeholder(lstm_4_while_less_lstm_4_strided_slice_1*
T0*
_output_shapes
: Y
lstm_4/while/IdentityIdentitylstm_4/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_4_while_identitylstm_4/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:
Ă

(__inference_dense_2_layer_call_fn_377253

inputs
unknown:	9
	unknown_0:9
identity˘StatefulPartitionedCallŘ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_372096o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ś
Ö
%sequential_2_lstm_4_while_body_370763D
@sequential_2_lstm_4_while_sequential_2_lstm_4_while_loop_counterJ
Fsequential_2_lstm_4_while_sequential_2_lstm_4_while_maximum_iterations)
%sequential_2_lstm_4_while_placeholder+
'sequential_2_lstm_4_while_placeholder_1+
'sequential_2_lstm_4_while_placeholder_2+
'sequential_2_lstm_4_while_placeholder_3C
?sequential_2_lstm_4_while_sequential_2_lstm_4_strided_slice_1_0
{sequential_2_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_4_tensorarrayunstack_tensorlistfromtensor_0X
Esequential_2_lstm_4_while_lstm_cell_4_split_readvariableop_resource_0:	9V
Gsequential_2_lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0:	S
?sequential_2_lstm_4_while_lstm_cell_4_readvariableop_resource_0:
&
"sequential_2_lstm_4_while_identity(
$sequential_2_lstm_4_while_identity_1(
$sequential_2_lstm_4_while_identity_2(
$sequential_2_lstm_4_while_identity_3(
$sequential_2_lstm_4_while_identity_4(
$sequential_2_lstm_4_while_identity_5A
=sequential_2_lstm_4_while_sequential_2_lstm_4_strided_slice_1}
ysequential_2_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_4_tensorarrayunstack_tensorlistfromtensorV
Csequential_2_lstm_4_while_lstm_cell_4_split_readvariableop_resource:	9T
Esequential_2_lstm_4_while_lstm_cell_4_split_1_readvariableop_resource:	Q
=sequential_2_lstm_4_while_lstm_cell_4_readvariableop_resource:
˘4sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp˘6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_1˘6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_2˘6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_3˘:sequential_2/lstm_4/while/lstm_cell_4/split/ReadVariableOp˘<sequential_2/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp
Ksequential_2/lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   
=sequential_2/lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_2_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_4_tensorarrayunstack_tensorlistfromtensor_0%sequential_2_lstm_4_while_placeholderTsequential_2/lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
element_dtype0
5sequential_2/lstm_4/while/lstm_cell_4/ones_like/ShapeShape'sequential_2_lstm_4_while_placeholder_2*
T0*
_output_shapes
:z
5sequential_2/lstm_4/while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ę
/sequential_2/lstm_4/while/lstm_cell_4/ones_likeFill>sequential_2/lstm_4/while/lstm_cell_4/ones_like/Shape:output:0>sequential_2/lstm_4/while/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
5sequential_2/lstm_4/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Á
:sequential_2/lstm_4/while/lstm_cell_4/split/ReadVariableOpReadVariableOpEsequential_2_lstm_4_while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	9*
dtype0
+sequential_2/lstm_4/while/lstm_cell_4/splitSplit>sequential_2/lstm_4/while/lstm_cell_4/split/split_dim:output:0Bsequential_2/lstm_4/while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_splitĺ
,sequential_2/lstm_4/while/lstm_cell_4/MatMulMatMulDsequential_2/lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_2/lstm_4/while/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_1MatMulDsequential_2/lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_2/lstm_4/while/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_2MatMulDsequential_2/lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_2/lstm_4/while/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_3MatMulDsequential_2/lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_2/lstm_4/while/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
7sequential_2/lstm_4/while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Á
<sequential_2/lstm_4/while/lstm_cell_4/split_1/ReadVariableOpReadVariableOpGsequential_2_lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0
-sequential_2/lstm_4/while/lstm_cell_4/split_1Split@sequential_2/lstm_4/while/lstm_cell_4/split_1/split_dim:output:0Dsequential_2/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_splitŰ
-sequential_2/lstm_4/while/lstm_cell_4/BiasAddBiasAdd6sequential_2/lstm_4/while/lstm_cell_4/MatMul:product:06sequential_2/lstm_4/while/lstm_cell_4/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ß
/sequential_2/lstm_4/while/lstm_cell_4/BiasAdd_1BiasAdd8sequential_2/lstm_4/while/lstm_cell_4/MatMul_1:product:06sequential_2/lstm_4/while/lstm_cell_4/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ß
/sequential_2/lstm_4/while/lstm_cell_4/BiasAdd_2BiasAdd8sequential_2/lstm_4/while/lstm_cell_4/MatMul_2:product:06sequential_2/lstm_4/while/lstm_cell_4/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ß
/sequential_2/lstm_4/while/lstm_cell_4/BiasAdd_3BiasAdd8sequential_2/lstm_4/while/lstm_cell_4/MatMul_3:product:06sequential_2/lstm_4/while/lstm_cell_4/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ć
)sequential_2/lstm_4/while/lstm_cell_4/mulMul'sequential_2_lstm_4_while_placeholder_28sequential_2/lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Č
+sequential_2/lstm_4/while/lstm_cell_4/mul_1Mul'sequential_2_lstm_4_while_placeholder_28sequential_2/lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Č
+sequential_2/lstm_4/while/lstm_cell_4/mul_2Mul'sequential_2_lstm_4_while_placeholder_28sequential_2/lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Č
+sequential_2/lstm_4/while/lstm_cell_4/mul_3Mul'sequential_2_lstm_4_while_placeholder_28sequential_2/lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ś
4sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOpReadVariableOp?sequential_2_lstm_4_while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
9sequential_2/lstm_4/while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
;sequential_2/lstm_4/while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
;sequential_2/lstm_4/while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ť
3sequential_2/lstm_4/while/lstm_cell_4/strided_sliceStridedSlice<sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp:value:0Bsequential_2/lstm_4/while/lstm_cell_4/strided_slice/stack:output:0Dsequential_2/lstm_4/while/lstm_cell_4/strided_slice/stack_1:output:0Dsequential_2/lstm_4/while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŘ
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_4MatMul-sequential_2/lstm_4/while/lstm_cell_4/mul:z:0<sequential_2/lstm_4/while/lstm_cell_4/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙×
)sequential_2/lstm_4/while/lstm_cell_4/addAddV26sequential_2/lstm_4/while/lstm_cell_4/BiasAdd:output:08sequential_2/lstm_4/while/lstm_cell_4/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
-sequential_2/lstm_4/while/lstm_cell_4/SigmoidSigmoid-sequential_2/lstm_4/while/lstm_cell_4/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_1ReadVariableOp?sequential_2_lstm_4_while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
;sequential_2/lstm_4/while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
=sequential_2/lstm_4/while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
=sequential_2/lstm_4/while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ľ
5sequential_2/lstm_4/while/lstm_cell_4/strided_slice_1StridedSlice>sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_1:value:0Dsequential_2/lstm_4/while/lstm_cell_4/strided_slice_1/stack:output:0Fsequential_2/lstm_4/while/lstm_cell_4/strided_slice_1/stack_1:output:0Fsequential_2/lstm_4/while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÜ
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_5MatMul/sequential_2/lstm_4/while/lstm_cell_4/mul_1:z:0>sequential_2/lstm_4/while/lstm_cell_4/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ű
+sequential_2/lstm_4/while/lstm_cell_4/add_1AddV28sequential_2/lstm_4/while/lstm_cell_4/BiasAdd_1:output:08sequential_2/lstm_4/while/lstm_cell_4/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
/sequential_2/lstm_4/while/lstm_cell_4/Sigmoid_1Sigmoid/sequential_2/lstm_4/while/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
+sequential_2/lstm_4/while/lstm_cell_4/mul_4Mul3sequential_2/lstm_4/while/lstm_cell_4/Sigmoid_1:y:0'sequential_2_lstm_4_while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_2ReadVariableOp?sequential_2_lstm_4_while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
;sequential_2/lstm_4/while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
=sequential_2/lstm_4/while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
=sequential_2/lstm_4/while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ľ
5sequential_2/lstm_4/while/lstm_cell_4/strided_slice_2StridedSlice>sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_2:value:0Dsequential_2/lstm_4/while/lstm_cell_4/strided_slice_2/stack:output:0Fsequential_2/lstm_4/while/lstm_cell_4/strided_slice_2/stack_1:output:0Fsequential_2/lstm_4/while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÜ
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_6MatMul/sequential_2/lstm_4/while/lstm_cell_4/mul_2:z:0>sequential_2/lstm_4/while/lstm_cell_4/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ű
+sequential_2/lstm_4/while/lstm_cell_4/add_2AddV28sequential_2/lstm_4/while/lstm_cell_4/BiasAdd_2:output:08sequential_2/lstm_4/while/lstm_cell_4/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
*sequential_2/lstm_4/while/lstm_cell_4/TanhTanh/sequential_2/lstm_4/while/lstm_cell_4/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Č
+sequential_2/lstm_4/while/lstm_cell_4/mul_5Mul1sequential_2/lstm_4/while/lstm_cell_4/Sigmoid:y:0.sequential_2/lstm_4/while/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙É
+sequential_2/lstm_4/while/lstm_cell_4/add_3AddV2/sequential_2/lstm_4/while/lstm_cell_4/mul_4:z:0/sequential_2/lstm_4/while/lstm_cell_4/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_3ReadVariableOp?sequential_2_lstm_4_while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
;sequential_2/lstm_4/while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
=sequential_2/lstm_4/while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
=sequential_2/lstm_4/while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ľ
5sequential_2/lstm_4/while/lstm_cell_4/strided_slice_3StridedSlice>sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_3:value:0Dsequential_2/lstm_4/while/lstm_cell_4/strided_slice_3/stack:output:0Fsequential_2/lstm_4/while/lstm_cell_4/strided_slice_3/stack_1:output:0Fsequential_2/lstm_4/while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÜ
.sequential_2/lstm_4/while/lstm_cell_4/MatMul_7MatMul/sequential_2/lstm_4/while/lstm_cell_4/mul_3:z:0>sequential_2/lstm_4/while/lstm_cell_4/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ű
+sequential_2/lstm_4/while/lstm_cell_4/add_4AddV28sequential_2/lstm_4/while/lstm_cell_4/BiasAdd_3:output:08sequential_2/lstm_4/while/lstm_cell_4/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
/sequential_2/lstm_4/while/lstm_cell_4/Sigmoid_2Sigmoid/sequential_2/lstm_4/while/lstm_cell_4/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
,sequential_2/lstm_4/while/lstm_cell_4/Tanh_1Tanh/sequential_2/lstm_4/while/lstm_cell_4/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ě
+sequential_2/lstm_4/while/lstm_cell_4/mul_6Mul3sequential_2/lstm_4/while/lstm_cell_4/Sigmoid_2:y:00sequential_2/lstm_4/while/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
>sequential_2/lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_2_lstm_4_while_placeholder_1%sequential_2_lstm_4_while_placeholder/sequential_2/lstm_4/while/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype0:éčŇa
sequential_2/lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_2/lstm_4/while/addAddV2%sequential_2_lstm_4_while_placeholder(sequential_2/lstm_4/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_2/lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ˇ
sequential_2/lstm_4/while/add_1AddV2@sequential_2_lstm_4_while_sequential_2_lstm_4_while_loop_counter*sequential_2/lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 
"sequential_2/lstm_4/while/IdentityIdentity#sequential_2/lstm_4/while/add_1:z:0^sequential_2/lstm_4/while/NoOp*
T0*
_output_shapes
: ş
$sequential_2/lstm_4/while/Identity_1IdentityFsequential_2_lstm_4_while_sequential_2_lstm_4_while_maximum_iterations^sequential_2/lstm_4/while/NoOp*
T0*
_output_shapes
: 
$sequential_2/lstm_4/while/Identity_2Identity!sequential_2/lstm_4/while/add:z:0^sequential_2/lstm_4/while/NoOp*
T0*
_output_shapes
: Ő
$sequential_2/lstm_4/while/Identity_3IdentityNsequential_2/lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_2/lstm_4/while/NoOp*
T0*
_output_shapes
: :éčŇľ
$sequential_2/lstm_4/while/Identity_4Identity/sequential_2/lstm_4/while/lstm_cell_4/mul_6:z:0^sequential_2/lstm_4/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ
$sequential_2/lstm_4/while/Identity_5Identity/sequential_2/lstm_4/while/lstm_cell_4/add_3:z:0^sequential_2/lstm_4/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ž
sequential_2/lstm_4/while/NoOpNoOp5^sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp7^sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_17^sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_27^sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_3;^sequential_2/lstm_4/while/lstm_cell_4/split/ReadVariableOp=^sequential_2/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Q
"sequential_2_lstm_4_while_identity+sequential_2/lstm_4/while/Identity:output:0"U
$sequential_2_lstm_4_while_identity_1-sequential_2/lstm_4/while/Identity_1:output:0"U
$sequential_2_lstm_4_while_identity_2-sequential_2/lstm_4/while/Identity_2:output:0"U
$sequential_2_lstm_4_while_identity_3-sequential_2/lstm_4/while/Identity_3:output:0"U
$sequential_2_lstm_4_while_identity_4-sequential_2/lstm_4/while/Identity_4:output:0"U
$sequential_2_lstm_4_while_identity_5-sequential_2/lstm_4/while/Identity_5:output:0"
=sequential_2_lstm_4_while_lstm_cell_4_readvariableop_resource?sequential_2_lstm_4_while_lstm_cell_4_readvariableop_resource_0"
Esequential_2_lstm_4_while_lstm_cell_4_split_1_readvariableop_resourceGsequential_2_lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0"
Csequential_2_lstm_4_while_lstm_cell_4_split_readvariableop_resourceEsequential_2_lstm_4_while_lstm_cell_4_split_readvariableop_resource_0"
=sequential_2_lstm_4_while_sequential_2_lstm_4_strided_slice_1?sequential_2_lstm_4_while_sequential_2_lstm_4_strided_slice_1_0"ř
ysequential_2_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_4_tensorarrayunstack_tensorlistfromtensor{sequential_2_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_2_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2l
4sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp4sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp2p
6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_16sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_12p
6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_26sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_22p
6sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_36sequential_2/lstm_4/while/lstm_cell_4/ReadVariableOp_32x
:sequential_2/lstm_4/while/lstm_cell_4/split/ReadVariableOp:sequential_2/lstm_4/while/lstm_cell_4/split/ReadVariableOp2|
<sequential_2/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp<sequential_2/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 
Ş
ˇ
'__inference_lstm_4_layer_call_fn_374585
inputs_0
unknown:	9
	unknown_0:	
	unknown_1:

identity˘StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_371328}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
"
_user_specified_name
inputs/0

Ą
3__inference_time_distributed_2_layer_call_fn_376759

inputs
unknown:	9
	unknown_0:9
identity˘StatefulPartitionedCallđ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_372107|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ş
Đ
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_372146

inputs!
dense_2_372136:	9
dense_2_372138:9
identity˘dense_2/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ö
dense_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_2_372136dense_2_372138*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_372096\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :9
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape(dense_2/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9h
NoOpNoOp ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
­
¸
'__inference_lstm_5_layer_call_fn_375673
inputs_0
unknown:

	unknown_0:	
	unknown_1:

identity˘StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_371796}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0
Â\
Ť
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_377028

inputs
states_0
states_10
split_readvariableop_resource:	9.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘split/ReadVariableOp˘split_1/ReadVariableOpG
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?q
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?u
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>­
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?u
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>­
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?u
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>­
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	9*
dtype0˘
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mulMulstates_0dropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
mul_1Mulstates_0dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
mul_2Mulstates_0dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
mul_3Mulstates_0dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskf
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙X
mul_4MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mul_5MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
IdentityIdentity	mul_6:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_1Identity	mul_6:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙9:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙9
 
_user_specified_nameinputs:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states/1
ę	
Ç
$__inference_signature_wrapper_374574
lstm_4_input
unknown:	9
	unknown_0:	
	unknown_1:

	unknown_2:

	unknown_3:	
	unknown_4:

	unknown_5:	9
	unknown_6:9
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalllstm_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_371135|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
&
_user_specified_namelstm_4_input
š
Ă
while_cond_372496
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_372496___redundant_placeholder04
0while_while_cond_372496___redundant_placeholder14
0while_while_cond_372496___redundant_placeholder24
0while_while_cond_372496___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:
š
Ă
while_cond_375502
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_375502___redundant_placeholder04
0while_while_cond_375502___redundant_placeholder14
0while_while_cond_375502___redundant_placeholder24
0while_while_cond_375502___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:
š
Ă
while_cond_375241
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_375241___redundant_placeholder04
0while_while_cond_375241___redundant_placeholder14
0while_while_cond_375241___redundant_placeholder24
0while_while_cond_375241___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:
ůl
	
while_body_372497
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_5_split_readvariableop_resource_0:
B
3while_lstm_cell_5_split_1_readvariableop_resource_0:	?
+while_lstm_cell_5_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_5_split_readvariableop_resource:
@
1while_lstm_cell_5_split_1_readvariableop_resource:	=
)while_lstm_cell_5_readvariableop_resource:
˘ while/lstm_cell_5/ReadVariableOp˘"while/lstm_cell_5/ReadVariableOp_1˘"while/lstm_cell_5/ReadVariableOp_2˘"while/lstm_cell_5/ReadVariableOp_3˘&while/lstm_cell_5/split/ReadVariableOp˘(while/lstm_cell_5/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0d
!while/lstm_cell_5/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell_5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ž
while/lstm_cell_5/ones_likeFill*while/lstm_cell_5/ones_like/Shape:output:0*while/lstm_cell_5/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_5/split/ReadVariableOpReadVariableOp1while_lstm_cell_5_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ü
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0.while/lstm_cell_5/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_splitŠ
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_5/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_5/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_5/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
#while/lstm_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_5/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_5_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Î
while/lstm_cell_5/split_1Split,while/lstm_cell_5/split_1/split_dim:output:00while/lstm_cell_5/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell_5/BiasAddBiasAdd"while/lstm_cell_5/MatMul:product:0"while/lstm_cell_5/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_5/BiasAdd_1BiasAdd$while/lstm_cell_5/MatMul_1:product:0"while/lstm_cell_5/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_5/BiasAdd_2BiasAdd$while/lstm_cell_5/MatMul_2:product:0"while/lstm_cell_5/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_5/BiasAdd_3BiasAdd$while/lstm_cell_5/MatMul_3:product:0"while/lstm_cell_5/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mulMulwhile_placeholder_2$while/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_1Mulwhile_placeholder_2$while/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_2Mulwhile_placeholder_2$while/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_3Mulwhile_placeholder_2$while/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_5/ReadVariableOpReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell_5/strided_sliceStridedSlice(while/lstm_cell_5/ReadVariableOp:value:0.while/lstm_cell_5/strided_slice/stack:output:00while/lstm_cell_5/strided_slice/stack_1:output:00while/lstm_cell_5/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_5/MatMul_4MatMulwhile/lstm_cell_5/mul:z:0(while/lstm_cell_5/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/addAddV2"while/lstm_cell_5/BiasAdd:output:0$while/lstm_cell_5/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell_5/SigmoidSigmoidwhile/lstm_cell_5/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_5/ReadVariableOp_1ReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_5/strided_slice_1StridedSlice*while/lstm_cell_5/ReadVariableOp_1:value:00while/lstm_cell_5/strided_slice_1/stack:output:02while/lstm_cell_5/strided_slice_1/stack_1:output:02while/lstm_cell_5/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_5/MatMul_5MatMulwhile/lstm_cell_5/mul_1:z:0*while/lstm_cell_5/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_1AddV2$while/lstm_cell_5/BiasAdd_1:output:0$while/lstm_cell_5/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_5/Sigmoid_1Sigmoidwhile/lstm_cell_5/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_4Mulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_5/ReadVariableOp_2ReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_5/strided_slice_2StridedSlice*while/lstm_cell_5/ReadVariableOp_2:value:00while/lstm_cell_5/strided_slice_2/stack:output:02while/lstm_cell_5/strided_slice_2/stack_1:output:02while/lstm_cell_5/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_5/MatMul_6MatMulwhile/lstm_cell_5/mul_2:z:0*while/lstm_cell_5/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_2AddV2$while/lstm_cell_5/BiasAdd_2:output:0$while/lstm_cell_5/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
while/lstm_cell_5/TanhTanhwhile/lstm_cell_5/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_5Mulwhile/lstm_cell_5/Sigmoid:y:0while/lstm_cell_5/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_3AddV2while/lstm_cell_5/mul_4:z:0while/lstm_cell_5/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_5/ReadVariableOp_3ReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_5/strided_slice_3StridedSlice*while/lstm_cell_5/ReadVariableOp_3:value:00while/lstm_cell_5/strided_slice_3/stack:output:02while/lstm_cell_5/strided_slice_3/stack_1:output:02while/lstm_cell_5/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_5/MatMul_7MatMulwhile/lstm_cell_5/mul_3:z:0*while/lstm_cell_5/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_4AddV2$while/lstm_cell_5/BiasAdd_3:output:0$while/lstm_cell_5/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_5/Sigmoid_2Sigmoidwhile/lstm_cell_5/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/lstm_cell_5/Tanh_1Tanhwhile/lstm_cell_5/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_6Mulwhile/lstm_cell_5/Sigmoid_2:y:0while/lstm_cell_5/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_6:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇy
while/Identity_4Identitywhile/lstm_cell_5/mul_6:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
while/Identity_5Identitywhile/lstm_cell_5/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˛

while/NoOpNoOp!^while/lstm_cell_5/ReadVariableOp#^while/lstm_cell_5/ReadVariableOp_1#^while/lstm_cell_5/ReadVariableOp_2#^while/lstm_cell_5/ReadVariableOp_3'^while/lstm_cell_5/split/ReadVariableOp)^while/lstm_cell_5/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_5_readvariableop_resource+while_lstm_cell_5_readvariableop_resource_0"h
1while_lstm_cell_5_split_1_readvariableop_resource3while_lstm_cell_5_split_1_readvariableop_resource_0"d
/while_lstm_cell_5_split_readvariableop_resource1while_lstm_cell_5_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2D
 while/lstm_cell_5/ReadVariableOp while/lstm_cell_5/ReadVariableOp2H
"while/lstm_cell_5/ReadVariableOp_1"while/lstm_cell_5/ReadVariableOp_12H
"while/lstm_cell_5/ReadVariableOp_2"while/lstm_cell_5/ReadVariableOp_22H
"while/lstm_cell_5/ReadVariableOp_3"while/lstm_cell_5/ReadVariableOp_32P
&while/lstm_cell_5/split/ReadVariableOp&while/lstm_cell_5/split/ReadVariableOp2T
(while/lstm_cell_5/split_1/ReadVariableOp(while/lstm_cell_5/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 
§
ś
'__inference_lstm_5_layer_call_fn_375695

inputs
unknown:

	unknown_0:	
	unknown_1:

identity˘StatefulPartitionedCallň
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_372624}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
˝
ú
H__inference_sequential_2_layer_call_and_return_conditional_losses_373402
lstm_4_input 
lstm_4_373380:	9
lstm_4_373382:	!
lstm_4_373384:
!
lstm_5_373387:

lstm_5_373389:	!
lstm_5_373391:
,
time_distributed_2_373394:	9'
time_distributed_2_373396:9
identity˘lstm_4/StatefulPartitionedCall˘lstm_5/StatefulPartitionedCall˘*time_distributed_2/StatefulPartitionedCall
lstm_4/StatefulPartitionedCallStatefulPartitionedCalllstm_4_inputlstm_4_373380lstm_4_373382lstm_4_373384*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_372388¨
lstm_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0lstm_5_373387lstm_5_373389lstm_5_373391*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_372624Ć
*time_distributed_2/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0time_distributed_2_373394time_distributed_2_373396*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_372107q
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ź
time_distributed_2/ReshapeReshape'lstm_5/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
IdentityIdentity3time_distributed_2/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9ľ
NoOpNoOp^lstm_4/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall+^time_distributed_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : : : : : : 2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall2X
*time_distributed_2/StatefulPartitionedCall*time_distributed_2/StatefulPartitionedCall:b ^
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
&
_user_specified_namelstm_4_input
ě"
Ţ
while_body_371259
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_4_371283_0:	9)
while_lstm_cell_4_371285_0:	.
while_lstm_cell_4_371287_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_4_371283:	9'
while_lstm_cell_4_371285:	,
while_lstm_cell_4_371287:
˘)while/lstm_cell_4/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
element_dtype0ą
)while/lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_4_371283_0while_lstm_cell_4_371285_0while_lstm_cell_4_371287_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_371245Ű
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇ
while/Identity_4Identity2while/lstm_cell_4/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/Identity_5Identity2while/lstm_cell_4/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x

while/NoOpNoOp*^while/lstm_cell_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_4_371283while_lstm_cell_4_371283_0"6
while_lstm_cell_4_371285while_lstm_cell_4_371285_0"6
while_lstm_cell_4_371287while_lstm_cell_4_371287_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2V
)while/lstm_cell_4/StatefulPartitionedCall)while/lstm_cell_4/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 
š
Ă
while_cond_376068
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_376068___redundant_placeholder04
0while_while_cond_376068___redundant_placeholder14
0while_while_cond_376068___redundant_placeholder24
0while_while_cond_376068___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:
ô
ö
,__inference_lstm_cell_4_layer_call_fn_376829

inputs
states_0
states_1
unknown:	9
	unknown_0:	
	unknown_1:

identity

identity_1

identity_2˘StatefulPartitionedCallŞ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_371245p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙9:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙9
 
_user_specified_nameinputs:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states/1
¨
í	
H__inference_sequential_2_layer_call_and_return_conditional_losses_373949

inputsC
0lstm_4_lstm_cell_4_split_readvariableop_resource:	9A
2lstm_4_lstm_cell_4_split_1_readvariableop_resource:	>
*lstm_4_lstm_cell_4_readvariableop_resource:
D
0lstm_5_lstm_cell_5_split_readvariableop_resource:
A
2lstm_5_lstm_cell_5_split_1_readvariableop_resource:	>
*lstm_5_lstm_cell_5_readvariableop_resource:
L
9time_distributed_2_dense_2_matmul_readvariableop_resource:	9H
:time_distributed_2_dense_2_biasadd_readvariableop_resource:9
identity˘!lstm_4/lstm_cell_4/ReadVariableOp˘#lstm_4/lstm_cell_4/ReadVariableOp_1˘#lstm_4/lstm_cell_4/ReadVariableOp_2˘#lstm_4/lstm_cell_4/ReadVariableOp_3˘'lstm_4/lstm_cell_4/split/ReadVariableOp˘)lstm_4/lstm_cell_4/split_1/ReadVariableOp˘lstm_4/while˘!lstm_5/lstm_cell_5/ReadVariableOp˘#lstm_5/lstm_cell_5/ReadVariableOp_1˘#lstm_5/lstm_cell_5/ReadVariableOp_2˘#lstm_5/lstm_cell_5/ReadVariableOp_3˘'lstm_5/lstm_cell_5/split/ReadVariableOp˘)lstm_5/lstm_cell_5/split_1/ReadVariableOp˘lstm_5/while˘1time_distributed_2/dense_2/BiasAdd/ReadVariableOp˘0time_distributed_2/dense_2/MatMul/ReadVariableOpB
lstm_4/ShapeShapeinputs*
T0*
_output_shapes
:d
lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_4/strided_sliceStridedSlicelstm_4/Shape:output:0#lstm_4/strided_slice/stack:output:0%lstm_4/strided_slice/stack_1:output:0%lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_4/zeros/packedPacklstm_4/strided_slice:output:0lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_4/zerosFilllstm_4/zeros/packed:output:0lstm_4/zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_4/zeros_1/packedPacklstm_4/strided_slice:output:0 lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_4/zeros_1Filllstm_4/zeros_1/packed:output:0lstm_4/zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_4/transpose	Transposeinputslstm_4/transpose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9R
lstm_4/Shape_1Shapelstm_4/transpose:y:0*
T0*
_output_shapes
:f
lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ţ
lstm_4/strided_slice_1StridedSlicelstm_4/Shape_1:output:0%lstm_4/strided_slice_1/stack:output:0'lstm_4/strided_slice_1/stack_1:output:0'lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙É
lstm_4/TensorArrayV2TensorListReserve+lstm_4/TensorArrayV2/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
<lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   ő
.lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_4/transpose:y:0Elstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇf
lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_4/strided_slice_2StridedSlicelstm_4/transpose:y:0%lstm_4/strided_slice_2/stack:output:0'lstm_4/strided_slice_2/stack_1:output:0'lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
shrink_axis_maskg
"lstm_4/lstm_cell_4/ones_like/ShapeShapelstm_4/zeros:output:0*
T0*
_output_shapes
:g
"lstm_4/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ą
lstm_4/lstm_cell_4/ones_likeFill+lstm_4/lstm_cell_4/ones_like/Shape:output:0+lstm_4/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
"lstm_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'lstm_4/lstm_cell_4/split/ReadVariableOpReadVariableOp0lstm_4_lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	9*
dtype0Ű
lstm_4/lstm_cell_4/splitSplit+lstm_4/lstm_cell_4/split/split_dim:output:0/lstm_4/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_split
lstm_4/lstm_cell_4/MatMulMatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/MatMul_1MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/MatMul_2MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/MatMul_3MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
$lstm_4/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
)lstm_4/lstm_cell_4/split_1/ReadVariableOpReadVariableOp2lstm_4_lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0Ń
lstm_4/lstm_cell_4/split_1Split-lstm_4/lstm_cell_4/split_1/split_dim:output:01lstm_4/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split˘
lstm_4/lstm_cell_4/BiasAddBiasAdd#lstm_4/lstm_cell_4/MatMul:product:0#lstm_4/lstm_cell_4/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
lstm_4/lstm_cell_4/BiasAdd_1BiasAdd%lstm_4/lstm_cell_4/MatMul_1:product:0#lstm_4/lstm_cell_4/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
lstm_4/lstm_cell_4/BiasAdd_2BiasAdd%lstm_4/lstm_cell_4/MatMul_2:product:0#lstm_4/lstm_cell_4/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
lstm_4/lstm_cell_4/BiasAdd_3BiasAdd%lstm_4/lstm_cell_4/MatMul_3:product:0#lstm_4/lstm_cell_4/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/mulMullstm_4/zeros:output:0%lstm_4/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/mul_1Mullstm_4/zeros:output:0%lstm_4/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/mul_2Mullstm_4/zeros:output:0%lstm_4/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/mul_3Mullstm_4/zeros:output:0%lstm_4/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!lstm_4/lstm_cell_4/ReadVariableOpReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0w
&lstm_4/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(lstm_4/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(lstm_4/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 lstm_4/lstm_cell_4/strided_sliceStridedSlice)lstm_4/lstm_cell_4/ReadVariableOp:value:0/lstm_4/lstm_cell_4/strided_slice/stack:output:01lstm_4/lstm_cell_4/strided_slice/stack_1:output:01lstm_4/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_4/lstm_cell_4/MatMul_4MatMullstm_4/lstm_cell_4/mul:z:0)lstm_4/lstm_cell_4/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/addAddV2#lstm_4/lstm_cell_4/BiasAdd:output:0%lstm_4/lstm_cell_4/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lstm_4/lstm_cell_4/SigmoidSigmoidlstm_4/lstm_cell_4/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#lstm_4/lstm_cell_4/ReadVariableOp_1ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0y
(lstm_4/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_4/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_4/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm_4/lstm_cell_4/strided_slice_1StridedSlice+lstm_4/lstm_cell_4/ReadVariableOp_1:value:01lstm_4/lstm_cell_4/strided_slice_1/stack:output:03lstm_4/lstm_cell_4/strided_slice_1/stack_1:output:03lstm_4/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŁ
lstm_4/lstm_cell_4/MatMul_5MatMullstm_4/lstm_cell_4/mul_1:z:0+lstm_4/lstm_cell_4/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_4/lstm_cell_4/add_1AddV2%lstm_4/lstm_cell_4/BiasAdd_1:output:0%lstm_4/lstm_cell_4/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_4/lstm_cell_4/Sigmoid_1Sigmoidlstm_4/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/mul_4Mul lstm_4/lstm_cell_4/Sigmoid_1:y:0lstm_4/zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#lstm_4/lstm_cell_4/ReadVariableOp_2ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0y
(lstm_4/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_4/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      {
*lstm_4/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm_4/lstm_cell_4/strided_slice_2StridedSlice+lstm_4/lstm_cell_4/ReadVariableOp_2:value:01lstm_4/lstm_cell_4/strided_slice_2/stack:output:03lstm_4/lstm_cell_4/strided_slice_2/stack_1:output:03lstm_4/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŁ
lstm_4/lstm_cell_4/MatMul_6MatMullstm_4/lstm_cell_4/mul_2:z:0+lstm_4/lstm_cell_4/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_4/lstm_cell_4/add_2AddV2%lstm_4/lstm_cell_4/BiasAdd_2:output:0%lstm_4/lstm_cell_4/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
lstm_4/lstm_cell_4/TanhTanhlstm_4/lstm_cell_4/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/mul_5Mullstm_4/lstm_cell_4/Sigmoid:y:0lstm_4/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/add_3AddV2lstm_4/lstm_cell_4/mul_4:z:0lstm_4/lstm_cell_4/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#lstm_4/lstm_cell_4/ReadVariableOp_3ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0y
(lstm_4/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      {
*lstm_4/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_4/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm_4/lstm_cell_4/strided_slice_3StridedSlice+lstm_4/lstm_cell_4/ReadVariableOp_3:value:01lstm_4/lstm_cell_4/strided_slice_3/stack:output:03lstm_4/lstm_cell_4/strided_slice_3/stack_1:output:03lstm_4/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŁ
lstm_4/lstm_cell_4/MatMul_7MatMullstm_4/lstm_cell_4/mul_3:z:0+lstm_4/lstm_cell_4/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_4/lstm_cell_4/add_4AddV2%lstm_4/lstm_cell_4/BiasAdd_3:output:0%lstm_4/lstm_cell_4/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_4/lstm_cell_4/Sigmoid_2Sigmoidlstm_4/lstm_cell_4/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
lstm_4/lstm_cell_4/Tanh_1Tanhlstm_4/lstm_cell_4/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/lstm_cell_4/mul_6Mul lstm_4/lstm_cell_4/Sigmoid_2:y:0lstm_4/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
$lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Í
lstm_4/TensorArrayV2_1TensorListReserve-lstm_4/TensorArrayV2_1/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇM
lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙[
lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ű
lstm_4/whileWhile"lstm_4/while/loop_counter:output:0(lstm_4/while/maximum_iterations:output:0lstm_4/time:output:0lstm_4/TensorArrayV2_1:handle:0lstm_4/zeros:output:0lstm_4/zeros_1:output:0lstm_4/strided_slice_1:output:0>lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_4_lstm_cell_4_split_readvariableop_resource2lstm_4_lstm_cell_4_split_1_readvariableop_resource*lstm_4_lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_4_while_body_373577*$
condR
lstm_4_while_cond_373576*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
7lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   á
)lstm_4/TensorArrayV2Stack/TensorListStackTensorListStacklstm_4/while:output:3@lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0o
lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙h
lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ť
lstm_4/strided_slice_3StridedSlice2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_4/strided_slice_3/stack:output:0'lstm_4/strided_slice_3/stack_1:output:0'lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskl
lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ľ
lstm_4/transpose_1	Transpose2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_4/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙b
lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    R
lstm_5/ShapeShapelstm_4/transpose_1:y:0*
T0*
_output_shapes
:d
lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_5/strided_sliceStridedSlicelstm_5/Shape:output:0#lstm_5/strided_slice/stack:output:0%lstm_5/strided_slice/stack_1:output:0%lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_5/zeros/packedPacklstm_5/strided_slice:output:0lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_5/zerosFilllstm_5/zeros/packed:output:0lstm_5/zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_5/zeros_1/packedPacklstm_5/strided_slice:output:0 lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_5/zeros_1Filllstm_5/zeros_1/packed:output:0lstm_5/zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_5/transpose	Transposelstm_4/transpose_1:y:0lstm_5/transpose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙R
lstm_5/Shape_1Shapelstm_5/transpose:y:0*
T0*
_output_shapes
:f
lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ţ
lstm_5/strided_slice_1StridedSlicelstm_5/Shape_1:output:0%lstm_5/strided_slice_1/stack:output:0'lstm_5/strided_slice_1/stack_1:output:0'lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙É
lstm_5/TensorArrayV2TensorListReserve+lstm_5/TensorArrayV2/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ő
.lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_5/transpose:y:0Elstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇf
lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_5/strided_slice_2StridedSlicelstm_5/transpose:y:0%lstm_5/strided_slice_2/stack:output:0'lstm_5/strided_slice_2/stack_1:output:0'lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskg
"lstm_5/lstm_cell_5/ones_like/ShapeShapelstm_5/zeros:output:0*
T0*
_output_shapes
:g
"lstm_5/lstm_cell_5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ą
lstm_5/lstm_cell_5/ones_likeFill+lstm_5/lstm_cell_5/ones_like/Shape:output:0+lstm_5/lstm_cell_5/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
"lstm_5/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'lstm_5/lstm_cell_5/split/ReadVariableOpReadVariableOp0lstm_5_lstm_cell_5_split_readvariableop_resource* 
_output_shapes
:
*
dtype0ß
lstm_5/lstm_cell_5/splitSplit+lstm_5/lstm_cell_5/split/split_dim:output:0/lstm_5/lstm_cell_5/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split
lstm_5/lstm_cell_5/MatMulMatMullstm_5/strided_slice_2:output:0!lstm_5/lstm_cell_5/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/MatMul_1MatMullstm_5/strided_slice_2:output:0!lstm_5/lstm_cell_5/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/MatMul_2MatMullstm_5/strided_slice_2:output:0!lstm_5/lstm_cell_5/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/MatMul_3MatMullstm_5/strided_slice_2:output:0!lstm_5/lstm_cell_5/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
$lstm_5/lstm_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
)lstm_5/lstm_cell_5/split_1/ReadVariableOpReadVariableOp2lstm_5_lstm_cell_5_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0Ń
lstm_5/lstm_cell_5/split_1Split-lstm_5/lstm_cell_5/split_1/split_dim:output:01lstm_5/lstm_cell_5/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split˘
lstm_5/lstm_cell_5/BiasAddBiasAdd#lstm_5/lstm_cell_5/MatMul:product:0#lstm_5/lstm_cell_5/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
lstm_5/lstm_cell_5/BiasAdd_1BiasAdd%lstm_5/lstm_cell_5/MatMul_1:product:0#lstm_5/lstm_cell_5/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
lstm_5/lstm_cell_5/BiasAdd_2BiasAdd%lstm_5/lstm_cell_5/MatMul_2:product:0#lstm_5/lstm_cell_5/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
lstm_5/lstm_cell_5/BiasAdd_3BiasAdd%lstm_5/lstm_cell_5/MatMul_3:product:0#lstm_5/lstm_cell_5/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/mulMullstm_5/zeros:output:0%lstm_5/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/mul_1Mullstm_5/zeros:output:0%lstm_5/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/mul_2Mullstm_5/zeros:output:0%lstm_5/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/mul_3Mullstm_5/zeros:output:0%lstm_5/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!lstm_5/lstm_cell_5/ReadVariableOpReadVariableOp*lstm_5_lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0w
&lstm_5/lstm_cell_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(lstm_5/lstm_cell_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(lstm_5/lstm_cell_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 lstm_5/lstm_cell_5/strided_sliceStridedSlice)lstm_5/lstm_cell_5/ReadVariableOp:value:0/lstm_5/lstm_cell_5/strided_slice/stack:output:01lstm_5/lstm_cell_5/strided_slice/stack_1:output:01lstm_5/lstm_cell_5/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_5/lstm_cell_5/MatMul_4MatMullstm_5/lstm_cell_5/mul:z:0)lstm_5/lstm_cell_5/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/addAddV2#lstm_5/lstm_cell_5/BiasAdd:output:0%lstm_5/lstm_cell_5/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lstm_5/lstm_cell_5/SigmoidSigmoidlstm_5/lstm_cell_5/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#lstm_5/lstm_cell_5/ReadVariableOp_1ReadVariableOp*lstm_5_lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0y
(lstm_5/lstm_cell_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_5/lstm_cell_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_5/lstm_cell_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm_5/lstm_cell_5/strided_slice_1StridedSlice+lstm_5/lstm_cell_5/ReadVariableOp_1:value:01lstm_5/lstm_cell_5/strided_slice_1/stack:output:03lstm_5/lstm_cell_5/strided_slice_1/stack_1:output:03lstm_5/lstm_cell_5/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŁ
lstm_5/lstm_cell_5/MatMul_5MatMullstm_5/lstm_cell_5/mul_1:z:0+lstm_5/lstm_cell_5/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_5/lstm_cell_5/add_1AddV2%lstm_5/lstm_cell_5/BiasAdd_1:output:0%lstm_5/lstm_cell_5/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_5/lstm_cell_5/Sigmoid_1Sigmoidlstm_5/lstm_cell_5/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/mul_4Mul lstm_5/lstm_cell_5/Sigmoid_1:y:0lstm_5/zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#lstm_5/lstm_cell_5/ReadVariableOp_2ReadVariableOp*lstm_5_lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0y
(lstm_5/lstm_cell_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_5/lstm_cell_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      {
*lstm_5/lstm_cell_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm_5/lstm_cell_5/strided_slice_2StridedSlice+lstm_5/lstm_cell_5/ReadVariableOp_2:value:01lstm_5/lstm_cell_5/strided_slice_2/stack:output:03lstm_5/lstm_cell_5/strided_slice_2/stack_1:output:03lstm_5/lstm_cell_5/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŁ
lstm_5/lstm_cell_5/MatMul_6MatMullstm_5/lstm_cell_5/mul_2:z:0+lstm_5/lstm_cell_5/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_5/lstm_cell_5/add_2AddV2%lstm_5/lstm_cell_5/BiasAdd_2:output:0%lstm_5/lstm_cell_5/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
lstm_5/lstm_cell_5/TanhTanhlstm_5/lstm_cell_5/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/mul_5Mullstm_5/lstm_cell_5/Sigmoid:y:0lstm_5/lstm_cell_5/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/add_3AddV2lstm_5/lstm_cell_5/mul_4:z:0lstm_5/lstm_cell_5/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#lstm_5/lstm_cell_5/ReadVariableOp_3ReadVariableOp*lstm_5_lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0y
(lstm_5/lstm_cell_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      {
*lstm_5/lstm_cell_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_5/lstm_cell_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm_5/lstm_cell_5/strided_slice_3StridedSlice+lstm_5/lstm_cell_5/ReadVariableOp_3:value:01lstm_5/lstm_cell_5/strided_slice_3/stack:output:03lstm_5/lstm_cell_5/strided_slice_3/stack_1:output:03lstm_5/lstm_cell_5/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŁ
lstm_5/lstm_cell_5/MatMul_7MatMullstm_5/lstm_cell_5/mul_3:z:0+lstm_5/lstm_cell_5/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_5/lstm_cell_5/add_4AddV2%lstm_5/lstm_cell_5/BiasAdd_3:output:0%lstm_5/lstm_cell_5/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_5/lstm_cell_5/Sigmoid_2Sigmoidlstm_5/lstm_cell_5/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
lstm_5/lstm_cell_5/Tanh_1Tanhlstm_5/lstm_cell_5/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/lstm_cell_5/mul_6Mul lstm_5/lstm_cell_5/Sigmoid_2:y:0lstm_5/lstm_cell_5/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
$lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Í
lstm_5/TensorArrayV2_1TensorListReserve-lstm_5/TensorArrayV2_1/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇM
lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙[
lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ű
lstm_5/whileWhile"lstm_5/while/loop_counter:output:0(lstm_5/while/maximum_iterations:output:0lstm_5/time:output:0lstm_5/TensorArrayV2_1:handle:0lstm_5/zeros:output:0lstm_5/zeros_1:output:0lstm_5/strided_slice_1:output:0>lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_5_lstm_cell_5_split_readvariableop_resource2lstm_5_lstm_cell_5_split_1_readvariableop_resource*lstm_5_lstm_cell_5_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_5_while_body_373802*$
condR
lstm_5_while_cond_373801*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   á
)lstm_5/TensorArrayV2Stack/TensorListStackTensorListStacklstm_5/while:output:3@lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0o
lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙h
lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ť
lstm_5/strided_slice_3StridedSlice2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_5/strided_slice_3/stack:output:0'lstm_5/strided_slice_3/stack_1:output:0'lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskl
lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ľ
lstm_5/transpose_1	Transpose2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_5/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙b
lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ^
time_distributed_2/ShapeShapelstm_5/transpose_1:y:0*
T0*
_output_shapes
:p
&time_distributed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(time_distributed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(time_distributed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 time_distributed_2/strided_sliceStridedSlice!time_distributed_2/Shape:output:0/time_distributed_2/strided_slice/stack:output:01time_distributed_2/strided_slice/stack_1:output:01time_distributed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
time_distributed_2/ReshapeReshapelstm_5/transpose_1:y:0)time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
0time_distributed_2/dense_2/MatMul/ReadVariableOpReadVariableOp9time_distributed_2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	9*
dtype0ź
!time_distributed_2/dense_2/MatMulMatMul#time_distributed_2/Reshape:output:08time_distributed_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9¨
1time_distributed_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp:time_distributed_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:9*
dtype0Ç
"time_distributed_2/dense_2/BiasAddBiasAdd+time_distributed_2/dense_2/MatMul:product:09time_distributed_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9
"time_distributed_2/dense_2/SoftmaxSoftmax+time_distributed_2/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9o
$time_distributed_2/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙f
$time_distributed_2/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :9á
"time_distributed_2/Reshape_1/shapePack-time_distributed_2/Reshape_1/shape/0:output:0)time_distributed_2/strided_slice:output:0-time_distributed_2/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:Á
time_distributed_2/Reshape_1Reshape,time_distributed_2/dense_2/Softmax:softmax:0+time_distributed_2/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9s
"time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
time_distributed_2/Reshape_2Reshapelstm_5/transpose_1:y:0+time_distributed_2/Reshape_2/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
IdentityIdentity%time_distributed_2/Reshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9Ł
NoOpNoOp"^lstm_4/lstm_cell_4/ReadVariableOp$^lstm_4/lstm_cell_4/ReadVariableOp_1$^lstm_4/lstm_cell_4/ReadVariableOp_2$^lstm_4/lstm_cell_4/ReadVariableOp_3(^lstm_4/lstm_cell_4/split/ReadVariableOp*^lstm_4/lstm_cell_4/split_1/ReadVariableOp^lstm_4/while"^lstm_5/lstm_cell_5/ReadVariableOp$^lstm_5/lstm_cell_5/ReadVariableOp_1$^lstm_5/lstm_cell_5/ReadVariableOp_2$^lstm_5/lstm_cell_5/ReadVariableOp_3(^lstm_5/lstm_cell_5/split/ReadVariableOp*^lstm_5/lstm_cell_5/split_1/ReadVariableOp^lstm_5/while2^time_distributed_2/dense_2/BiasAdd/ReadVariableOp1^time_distributed_2/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : : : : : : 2F
!lstm_4/lstm_cell_4/ReadVariableOp!lstm_4/lstm_cell_4/ReadVariableOp2J
#lstm_4/lstm_cell_4/ReadVariableOp_1#lstm_4/lstm_cell_4/ReadVariableOp_12J
#lstm_4/lstm_cell_4/ReadVariableOp_2#lstm_4/lstm_cell_4/ReadVariableOp_22J
#lstm_4/lstm_cell_4/ReadVariableOp_3#lstm_4/lstm_cell_4/ReadVariableOp_32R
'lstm_4/lstm_cell_4/split/ReadVariableOp'lstm_4/lstm_cell_4/split/ReadVariableOp2V
)lstm_4/lstm_cell_4/split_1/ReadVariableOp)lstm_4/lstm_cell_4/split_1/ReadVariableOp2
lstm_4/whilelstm_4/while2F
!lstm_5/lstm_cell_5/ReadVariableOp!lstm_5/lstm_cell_5/ReadVariableOp2J
#lstm_5/lstm_cell_5/ReadVariableOp_1#lstm_5/lstm_cell_5/ReadVariableOp_12J
#lstm_5/lstm_cell_5/ReadVariableOp_2#lstm_5/lstm_cell_5/ReadVariableOp_22J
#lstm_5/lstm_cell_5/ReadVariableOp_3#lstm_5/lstm_cell_5/ReadVariableOp_32R
'lstm_5/lstm_cell_5/split/ReadVariableOp'lstm_5/lstm_cell_5/split/ReadVariableOp2V
)lstm_5/lstm_cell_5/split_1/ReadVariableOp)lstm_5/lstm_cell_5/split_1/ReadVariableOp2
lstm_5/whilelstm_5/while2f
1time_distributed_2/dense_2/BiasAdd/ReadVariableOp1time_distributed_2/dense_2/BiasAdd/ReadVariableOp2d
0time_distributed_2/dense_2/MatMul/ReadVariableOp0time_distributed_2/dense_2/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
 
_user_specified_nameinputs
z
ß
B__inference_lstm_5_layer_call_and_return_conditional_losses_372624

inputs=
)lstm_cell_5_split_readvariableop_resource:
:
+lstm_cell_5_split_1_readvariableop_resource:	7
#lstm_cell_5_readvariableop_resource:

identity˘lstm_cell_5/ReadVariableOp˘lstm_cell_5/ReadVariableOp_1˘lstm_cell_5/ReadVariableOp_2˘lstm_cell_5/ReadVariableOp_3˘ lstm_cell_5/split/ReadVariableOp˘"lstm_cell_5/split_1/ReadVariableOp˘while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ę
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskY
lstm_cell_5/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell_5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_5/ones_likeFill$lstm_cell_5/ones_like/Shape:output:0$lstm_cell_5/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_5/split/ReadVariableOpReadVariableOp)lstm_cell_5_split_readvariableop_resource* 
_output_shapes
:
*
dtype0Ę
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0(lstm_cell_5/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0lstm_cell_5/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_5/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_5/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_5/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
lstm_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_5/split_1/ReadVariableOpReadVariableOp+lstm_cell_5_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ź
lstm_cell_5/split_1Split&lstm_cell_5/split_1/split_dim:output:0*lstm_cell_5/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell_5/BiasAddBiasAddlstm_cell_5/MatMul:product:0lstm_cell_5/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/BiasAdd_1BiasAddlstm_cell_5/MatMul_1:product:0lstm_cell_5/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/BiasAdd_2BiasAddlstm_cell_5/MatMul_2:product:0lstm_cell_5/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/BiasAdd_3BiasAddlstm_cell_5/MatMul_3:product:0lstm_cell_5/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
lstm_cell_5/mulMulzeros:output:0lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_5/mul_1Mulzeros:output:0lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_5/mul_2Mulzeros:output:0lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_5/mul_3Mulzeros:output:0lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOpReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell_5/strided_sliceStridedSlice"lstm_cell_5/ReadVariableOp:value:0(lstm_cell_5/strided_slice/stack:output:0*lstm_cell_5/strided_slice/stack_1:output:0*lstm_cell_5/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_4MatMullstm_cell_5/mul:z:0"lstm_cell_5/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/addAddV2lstm_cell_5/BiasAdd:output:0lstm_cell_5/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell_5/SigmoidSigmoidlstm_cell_5/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOp_1ReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_5/strided_slice_1StridedSlice$lstm_cell_5/ReadVariableOp_1:value:0*lstm_cell_5/strided_slice_1/stack:output:0,lstm_cell_5/strided_slice_1/stack_1:output:0,lstm_cell_5/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_5MatMullstm_cell_5/mul_1:z:0$lstm_cell_5/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/add_1AddV2lstm_cell_5/BiasAdd_1:output:0lstm_cell_5/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_cell_5/mul_4Mullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOp_2ReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_5/strided_slice_2StridedSlice$lstm_cell_5/ReadVariableOp_2:value:0*lstm_cell_5/strided_slice_2/stack:output:0,lstm_cell_5/strided_slice_2/stack_1:output:0,lstm_cell_5/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_6MatMullstm_cell_5/mul_2:z:0$lstm_cell_5/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/add_2AddV2lstm_cell_5/BiasAdd_2:output:0lstm_cell_5/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
lstm_cell_5/TanhTanhlstm_cell_5/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
lstm_cell_5/mul_5Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_5/add_3AddV2lstm_cell_5/mul_4:z:0lstm_cell_5/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOp_3ReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_5/strided_slice_3StridedSlice$lstm_cell_5/ReadVariableOp_3:value:0*lstm_cell_5/strided_slice_3/stack:output:0,lstm_cell_5/strided_slice_3/stack_1:output:0,lstm_cell_5/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_7MatMullstm_cell_5/mul_3:z:0$lstm_cell_5/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/add_4AddV2lstm_cell_5/BiasAdd_3:output:0lstm_cell_5/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
lstm_cell_5/Tanh_1Tanhlstm_cell_5/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell_5/mul_6Mullstm_cell_5/Sigmoid_2:y:0lstm_cell_5/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ů
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_5_split_readvariableop_resource+lstm_cell_5_split_1_readvariableop_resource#lstm_cell_5_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_372497*
condR
while_cond_372496*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp^lstm_cell_5/ReadVariableOp^lstm_cell_5/ReadVariableOp_1^lstm_cell_5/ReadVariableOp_2^lstm_cell_5/ReadVariableOp_3!^lstm_cell_5/split/ReadVariableOp#^lstm_cell_5/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 28
lstm_cell_5/ReadVariableOplstm_cell_5/ReadVariableOp2<
lstm_cell_5/ReadVariableOp_1lstm_cell_5/ReadVariableOp_12<
lstm_cell_5/ReadVariableOp_2lstm_cell_5/ReadVariableOp_22<
lstm_cell_5/ReadVariableOp_3lstm_cell_5/ReadVariableOp_32D
 lstm_cell_5/split/ReadVariableOp lstm_cell_5/split/ReadVariableOp2H
"lstm_cell_5/split_1/ReadVariableOp"lstm_cell_5/split_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
8

B__inference_lstm_4_layer_call_and_return_conditional_losses_371328

inputs%
lstm_cell_4_371246:	9!
lstm_cell_4_371248:	&
lstm_cell_4_371250:

identity˘#lstm_cell_4/StatefulPartitionedCall˘while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
shrink_axis_maskó
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4_371246lstm_cell_4_371248lstm_cell_4_371250*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_371245n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¸
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4_371246lstm_cell_4_371248lstm_cell_4_371250*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_371259*
condR
while_cond_371258*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙t
NoOpNoOp$^lstm_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : 2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
 
_user_specified_nameinputs
Ú
ŕ
B__inference_lstm_4_layer_call_and_return_conditional_losses_375140
inputs_0<
)lstm_cell_4_split_readvariableop_resource:	9:
+lstm_cell_4_split_1_readvariableop_resource:	7
#lstm_cell_4_readvariableop_resource:

identity˘lstm_cell_4/ReadVariableOp˘lstm_cell_4/ReadVariableOp_1˘lstm_cell_4/ReadVariableOp_2˘lstm_cell_4/ReadVariableOp_3˘ lstm_cell_4/split/ReadVariableOp˘"lstm_cell_4/split_1/ReadVariableOp˘while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
shrink_axis_maskY
lstm_cell_4/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_4/dropout/MulMullstm_cell_4/ones_like:output:0"lstm_cell_4/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
lstm_cell_4/dropout/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:Ľ
0lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_4/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ë
 lstm_cell_4/dropout/GreaterEqualGreaterEqual9lstm_cell_4/dropout/random_uniform/RandomUniform:output:0+lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout/CastCast$lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout/Mul_1Mullstm_cell_4/dropout/Mul:z:0lstm_cell_4/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_4/dropout_1/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
lstm_cell_4/dropout_1/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:Š
2lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0i
$lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ń
"lstm_cell_4/dropout_1/GreaterEqualGreaterEqual;lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout_1/CastCast&lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout_1/Mul_1Mullstm_cell_4/dropout_1/Mul:z:0lstm_cell_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_4/dropout_2/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
lstm_cell_4/dropout_2/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:Š
2lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0i
$lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ń
"lstm_cell_4/dropout_2/GreaterEqualGreaterEqual;lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout_2/CastCast&lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout_2/Mul_1Mullstm_cell_4/dropout_2/Mul:z:0lstm_cell_4/dropout_2/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_4/dropout_3/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
lstm_cell_4/dropout_3/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:Š
2lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0i
$lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ń
"lstm_cell_4/dropout_3/GreaterEqualGreaterEqual;lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout_3/CastCast&lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout_3/Mul_1Mullstm_cell_4/dropout_3/Mul:z:0lstm_cell_4/dropout_3/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	9*
dtype0Ć
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_split
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ź
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/dropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
lstm_cell_4/TanhTanhlstm_cell_4/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
lstm_cell_4/Tanh_1Tanhlstm_cell_4/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ů
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_374981*
condR
while_cond_374980*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : 28
lstm_cell_4/ReadVariableOplstm_cell_4/ReadVariableOp2<
lstm_cell_4/ReadVariableOp_1lstm_cell_4/ReadVariableOp_12<
lstm_cell_4/ReadVariableOp_2lstm_cell_4/ReadVariableOp_22<
lstm_cell_4/ReadVariableOp_3lstm_cell_4/ReadVariableOp_32D
 lstm_cell_4/split/ReadVariableOp lstm_cell_4/split/ReadVariableOp2H
"lstm_cell_4/split_1/ReadVariableOp"lstm_cell_4/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
"
_user_specified_name
inputs/0
ůl
	
while_body_375808
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_5_split_readvariableop_resource_0:
B
3while_lstm_cell_5_split_1_readvariableop_resource_0:	?
+while_lstm_cell_5_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_5_split_readvariableop_resource:
@
1while_lstm_cell_5_split_1_readvariableop_resource:	=
)while_lstm_cell_5_readvariableop_resource:
˘ while/lstm_cell_5/ReadVariableOp˘"while/lstm_cell_5/ReadVariableOp_1˘"while/lstm_cell_5/ReadVariableOp_2˘"while/lstm_cell_5/ReadVariableOp_3˘&while/lstm_cell_5/split/ReadVariableOp˘(while/lstm_cell_5/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0d
!while/lstm_cell_5/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell_5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ž
while/lstm_cell_5/ones_likeFill*while/lstm_cell_5/ones_like/Shape:output:0*while/lstm_cell_5/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_5/split/ReadVariableOpReadVariableOp1while_lstm_cell_5_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ü
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0.while/lstm_cell_5/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_splitŠ
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_5/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_5/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_5/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
#while/lstm_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_5/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_5_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Î
while/lstm_cell_5/split_1Split,while/lstm_cell_5/split_1/split_dim:output:00while/lstm_cell_5/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell_5/BiasAddBiasAdd"while/lstm_cell_5/MatMul:product:0"while/lstm_cell_5/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_5/BiasAdd_1BiasAdd$while/lstm_cell_5/MatMul_1:product:0"while/lstm_cell_5/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_5/BiasAdd_2BiasAdd$while/lstm_cell_5/MatMul_2:product:0"while/lstm_cell_5/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_5/BiasAdd_3BiasAdd$while/lstm_cell_5/MatMul_3:product:0"while/lstm_cell_5/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mulMulwhile_placeholder_2$while/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_1Mulwhile_placeholder_2$while/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_2Mulwhile_placeholder_2$while/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_3Mulwhile_placeholder_2$while/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_5/ReadVariableOpReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell_5/strided_sliceStridedSlice(while/lstm_cell_5/ReadVariableOp:value:0.while/lstm_cell_5/strided_slice/stack:output:00while/lstm_cell_5/strided_slice/stack_1:output:00while/lstm_cell_5/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_5/MatMul_4MatMulwhile/lstm_cell_5/mul:z:0(while/lstm_cell_5/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/addAddV2"while/lstm_cell_5/BiasAdd:output:0$while/lstm_cell_5/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell_5/SigmoidSigmoidwhile/lstm_cell_5/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_5/ReadVariableOp_1ReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_5/strided_slice_1StridedSlice*while/lstm_cell_5/ReadVariableOp_1:value:00while/lstm_cell_5/strided_slice_1/stack:output:02while/lstm_cell_5/strided_slice_1/stack_1:output:02while/lstm_cell_5/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_5/MatMul_5MatMulwhile/lstm_cell_5/mul_1:z:0*while/lstm_cell_5/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_1AddV2$while/lstm_cell_5/BiasAdd_1:output:0$while/lstm_cell_5/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_5/Sigmoid_1Sigmoidwhile/lstm_cell_5/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_4Mulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_5/ReadVariableOp_2ReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_5/strided_slice_2StridedSlice*while/lstm_cell_5/ReadVariableOp_2:value:00while/lstm_cell_5/strided_slice_2/stack:output:02while/lstm_cell_5/strided_slice_2/stack_1:output:02while/lstm_cell_5/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_5/MatMul_6MatMulwhile/lstm_cell_5/mul_2:z:0*while/lstm_cell_5/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_2AddV2$while/lstm_cell_5/BiasAdd_2:output:0$while/lstm_cell_5/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
while/lstm_cell_5/TanhTanhwhile/lstm_cell_5/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_5Mulwhile/lstm_cell_5/Sigmoid:y:0while/lstm_cell_5/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_3AddV2while/lstm_cell_5/mul_4:z:0while/lstm_cell_5/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_5/ReadVariableOp_3ReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_5/strided_slice_3StridedSlice*while/lstm_cell_5/ReadVariableOp_3:value:00while/lstm_cell_5/strided_slice_3/stack:output:02while/lstm_cell_5/strided_slice_3/stack_1:output:02while/lstm_cell_5/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_5/MatMul_7MatMulwhile/lstm_cell_5/mul_3:z:0*while/lstm_cell_5/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_4AddV2$while/lstm_cell_5/BiasAdd_3:output:0$while/lstm_cell_5/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_5/Sigmoid_2Sigmoidwhile/lstm_cell_5/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/lstm_cell_5/Tanh_1Tanhwhile/lstm_cell_5/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_6Mulwhile/lstm_cell_5/Sigmoid_2:y:0while/lstm_cell_5/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_6:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇy
while/Identity_4Identitywhile/lstm_cell_5/mul_6:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
while/Identity_5Identitywhile/lstm_cell_5/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˛

while/NoOpNoOp!^while/lstm_cell_5/ReadVariableOp#^while/lstm_cell_5/ReadVariableOp_1#^while/lstm_cell_5/ReadVariableOp_2#^while/lstm_cell_5/ReadVariableOp_3'^while/lstm_cell_5/split/ReadVariableOp)^while/lstm_cell_5/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_5_readvariableop_resource+while_lstm_cell_5_readvariableop_resource_0"h
1while_lstm_cell_5_split_1_readvariableop_resource3while_lstm_cell_5_split_1_readvariableop_resource_0"d
/while_lstm_cell_5_split_readvariableop_resource1while_lstm_cell_5_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2D
 while/lstm_cell_5/ReadVariableOp while/lstm_cell_5/ReadVariableOp2H
"while/lstm_cell_5/ReadVariableOp_1"while/lstm_cell_5/ReadVariableOp_12H
"while/lstm_cell_5/ReadVariableOp_2"while/lstm_cell_5/ReadVariableOp_22H
"while/lstm_cell_5/ReadVariableOp_3"while/lstm_cell_5/ReadVariableOp_32P
&while/lstm_cell_5/split/ReadVariableOp&while/lstm_cell_5/split/ReadVariableOp2T
(while/lstm_cell_5/split_1/ReadVariableOp(while/lstm_cell_5/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 
z
á
B__inference_lstm_5_layer_call_and_return_conditional_losses_375935
inputs_0=
)lstm_cell_5_split_readvariableop_resource:
:
+lstm_cell_5_split_1_readvariableop_resource:	7
#lstm_cell_5_readvariableop_resource:

identity˘lstm_cell_5/ReadVariableOp˘lstm_cell_5/ReadVariableOp_1˘lstm_cell_5/ReadVariableOp_2˘lstm_cell_5/ReadVariableOp_3˘ lstm_cell_5/split/ReadVariableOp˘"lstm_cell_5/split_1/ReadVariableOp˘while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ę
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskY
lstm_cell_5/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell_5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_5/ones_likeFill$lstm_cell_5/ones_like/Shape:output:0$lstm_cell_5/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_5/split/ReadVariableOpReadVariableOp)lstm_cell_5_split_readvariableop_resource* 
_output_shapes
:
*
dtype0Ę
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0(lstm_cell_5/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0lstm_cell_5/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_5/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_5/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_5/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
lstm_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_5/split_1/ReadVariableOpReadVariableOp+lstm_cell_5_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ź
lstm_cell_5/split_1Split&lstm_cell_5/split_1/split_dim:output:0*lstm_cell_5/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell_5/BiasAddBiasAddlstm_cell_5/MatMul:product:0lstm_cell_5/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/BiasAdd_1BiasAddlstm_cell_5/MatMul_1:product:0lstm_cell_5/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/BiasAdd_2BiasAddlstm_cell_5/MatMul_2:product:0lstm_cell_5/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/BiasAdd_3BiasAddlstm_cell_5/MatMul_3:product:0lstm_cell_5/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
lstm_cell_5/mulMulzeros:output:0lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_5/mul_1Mulzeros:output:0lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_5/mul_2Mulzeros:output:0lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_5/mul_3Mulzeros:output:0lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOpReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell_5/strided_sliceStridedSlice"lstm_cell_5/ReadVariableOp:value:0(lstm_cell_5/strided_slice/stack:output:0*lstm_cell_5/strided_slice/stack_1:output:0*lstm_cell_5/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_4MatMullstm_cell_5/mul:z:0"lstm_cell_5/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/addAddV2lstm_cell_5/BiasAdd:output:0lstm_cell_5/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell_5/SigmoidSigmoidlstm_cell_5/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOp_1ReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_5/strided_slice_1StridedSlice$lstm_cell_5/ReadVariableOp_1:value:0*lstm_cell_5/strided_slice_1/stack:output:0,lstm_cell_5/strided_slice_1/stack_1:output:0,lstm_cell_5/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_5MatMullstm_cell_5/mul_1:z:0$lstm_cell_5/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/add_1AddV2lstm_cell_5/BiasAdd_1:output:0lstm_cell_5/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_cell_5/mul_4Mullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOp_2ReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_5/strided_slice_2StridedSlice$lstm_cell_5/ReadVariableOp_2:value:0*lstm_cell_5/strided_slice_2/stack:output:0,lstm_cell_5/strided_slice_2/stack_1:output:0,lstm_cell_5/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_6MatMullstm_cell_5/mul_2:z:0$lstm_cell_5/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/add_2AddV2lstm_cell_5/BiasAdd_2:output:0lstm_cell_5/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
lstm_cell_5/TanhTanhlstm_cell_5/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
lstm_cell_5/mul_5Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_5/add_3AddV2lstm_cell_5/mul_4:z:0lstm_cell_5/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOp_3ReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_5/strided_slice_3StridedSlice$lstm_cell_5/ReadVariableOp_3:value:0*lstm_cell_5/strided_slice_3/stack:output:0,lstm_cell_5/strided_slice_3/stack_1:output:0,lstm_cell_5/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_7MatMullstm_cell_5/mul_3:z:0$lstm_cell_5/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/add_4AddV2lstm_cell_5/BiasAdd_3:output:0lstm_cell_5/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
lstm_cell_5/Tanh_1Tanhlstm_cell_5/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell_5/mul_6Mullstm_cell_5/Sigmoid_2:y:0lstm_cell_5/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ů
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_5_split_readvariableop_resource+lstm_cell_5_split_1_readvariableop_resource#lstm_cell_5_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_375808*
condR
while_cond_375807*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp^lstm_cell_5/ReadVariableOp^lstm_cell_5/ReadVariableOp_1^lstm_cell_5/ReadVariableOp_2^lstm_cell_5/ReadVariableOp_3!^lstm_cell_5/split/ReadVariableOp#^lstm_cell_5/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 28
lstm_cell_5/ReadVariableOplstm_cell_5/ReadVariableOp2<
lstm_cell_5/ReadVariableOp_1lstm_cell_5/ReadVariableOp_12<
lstm_cell_5/ReadVariableOp_2lstm_cell_5/ReadVariableOp_22<
lstm_cell_5/ReadVariableOp_3lstm_cell_5/ReadVariableOp_32D
 lstm_cell_5/split/ReadVariableOp lstm_cell_5/split/ReadVariableOp2H
"lstm_cell_5/split_1/ReadVariableOp"lstm_cell_5/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0
Ň
Ţ
B__inference_lstm_4_layer_call_and_return_conditional_losses_375662

inputs<
)lstm_cell_4_split_readvariableop_resource:	9:
+lstm_cell_4_split_1_readvariableop_resource:	7
#lstm_cell_4_readvariableop_resource:

identity˘lstm_cell_4/ReadVariableOp˘lstm_cell_4/ReadVariableOp_1˘lstm_cell_4/ReadVariableOp_2˘lstm_cell_4/ReadVariableOp_3˘ lstm_cell_4/split/ReadVariableOp˘"lstm_cell_4/split_1/ReadVariableOp˘while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
shrink_axis_maskY
lstm_cell_4/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_4/dropout/MulMullstm_cell_4/ones_like:output:0"lstm_cell_4/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
lstm_cell_4/dropout/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:Ľ
0lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_4/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ë
 lstm_cell_4/dropout/GreaterEqualGreaterEqual9lstm_cell_4/dropout/random_uniform/RandomUniform:output:0+lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout/CastCast$lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout/Mul_1Mullstm_cell_4/dropout/Mul:z:0lstm_cell_4/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_4/dropout_1/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
lstm_cell_4/dropout_1/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:Š
2lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0i
$lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ń
"lstm_cell_4/dropout_1/GreaterEqualGreaterEqual;lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout_1/CastCast&lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout_1/Mul_1Mullstm_cell_4/dropout_1/Mul:z:0lstm_cell_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_4/dropout_2/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
lstm_cell_4/dropout_2/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:Š
2lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0i
$lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ń
"lstm_cell_4/dropout_2/GreaterEqualGreaterEqual;lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout_2/CastCast&lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout_2/Mul_1Mullstm_cell_4/dropout_2/Mul:z:0lstm_cell_4/dropout_2/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_4/dropout_3/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
lstm_cell_4/dropout_3/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:Š
2lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0i
$lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ń
"lstm_cell_4/dropout_3/GreaterEqualGreaterEqual;lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout_3/CastCast&lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/dropout_3/Mul_1Mullstm_cell_4/dropout_3/Mul:z:0lstm_cell_4/dropout_3/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	9*
dtype0Ć
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_split
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ź
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/dropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
lstm_cell_4/TanhTanhlstm_cell_4/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
lstm_cell_4/Tanh_1Tanhlstm_cell_4/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ů
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_375503*
condR
while_cond_375502*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : 28
lstm_cell_4/ReadVariableOplstm_cell_4/ReadVariableOp2<
lstm_cell_4/ReadVariableOp_1lstm_cell_4/ReadVariableOp_12<
lstm_cell_4/ReadVariableOp_2lstm_cell_4/ReadVariableOp_22<
lstm_cell_4/ReadVariableOp_3lstm_cell_4/ReadVariableOp_32D
 lstm_cell_4/split/ReadVariableOp lstm_cell_4/split/ReadVariableOp2H
"lstm_cell_4/split_1/ReadVariableOp"lstm_cell_4/split_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
 
_user_specified_nameinputs
ş\
Ş
G__inference_lstm_cell_5_layer_call_and_return_conditional_losses_371934

inputs

states
states_11
split_readvariableop_resource:
.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘split/ReadVariableOp˘split_1/ReadVariableOpE
ones_like/ShapeShapestates*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?q
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?u
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>­
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?u
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>­
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?u
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>­
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
*
dtype0Ś
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙X
mulMulstatesdropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
mul_1Mulstatesdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
mul_2Mulstatesdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
mul_3Mulstatesdropout_3/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskf
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙X
mul_4MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mul_5MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
IdentityIdentity	mul_6:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_1Identity	mul_6:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namestates:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namestates
Ť
ô
H__inference_sequential_2_layer_call_and_return_conditional_losses_373337

inputs 
lstm_4_373315:	9
lstm_4_373317:	!
lstm_4_373319:
!
lstm_5_373322:

lstm_5_373324:	!
lstm_5_373326:
,
time_distributed_2_373329:	9'
time_distributed_2_373331:9
identity˘lstm_4/StatefulPartitionedCall˘lstm_5/StatefulPartitionedCall˘*time_distributed_2/StatefulPartitionedCall
lstm_4/StatefulPartitionedCallStatefulPartitionedCallinputslstm_4_373315lstm_4_373317lstm_4_373319*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_373280¨
lstm_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0lstm_5_373322lstm_5_373324lstm_5_373326*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_372965Ć
*time_distributed_2/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0time_distributed_2_373329time_distributed_2_373331*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_372146q
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ź
time_distributed_2/ReshapeReshape'lstm_5/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
IdentityIdentity3time_distributed_2/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9ľ
NoOpNoOp^lstm_4/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall+^time_distributed_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : : : : : : 2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall2X
*time_distributed_2/StatefulPartitionedCall*time_distributed_2/StatefulPartitionedCall:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
 
_user_specified_nameinputs
z
Ţ
B__inference_lstm_4_layer_call_and_return_conditional_losses_372388

inputs<
)lstm_cell_4_split_readvariableop_resource:	9:
+lstm_cell_4_split_1_readvariableop_resource:	7
#lstm_cell_4_readvariableop_resource:

identity˘lstm_cell_4/ReadVariableOp˘lstm_cell_4/ReadVariableOp_1˘lstm_cell_4/ReadVariableOp_2˘lstm_cell_4/ReadVariableOp_3˘ lstm_cell_4/split/ReadVariableOp˘"lstm_cell_4/split_1/ReadVariableOp˘while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
shrink_axis_maskY
lstm_cell_4/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	9*
dtype0Ć
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_split
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ź
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
lstm_cell_4/TanhTanhlstm_cell_4/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
lstm_cell_4/Tanh_1Tanhlstm_cell_4/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ů
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_372261*
condR
while_cond_372260*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : 28
lstm_cell_4/ReadVariableOplstm_cell_4/ReadVariableOp2<
lstm_cell_4/ReadVariableOp_1lstm_cell_4/ReadVariableOp_12<
lstm_cell_4/ReadVariableOp_2lstm_cell_4/ReadVariableOp_22<
lstm_cell_4/ReadVariableOp_3lstm_cell_4/ReadVariableOp_32D
 lstm_cell_4/split/ReadVariableOp lstm_cell_4/split/ReadVariableOp2H
"lstm_cell_4/split_1/ReadVariableOp"lstm_cell_4/split_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
 
_user_specified_nameinputs
˝
ú
H__inference_sequential_2_layer_call_and_return_conditional_losses_373427
lstm_4_input 
lstm_4_373405:	9
lstm_4_373407:	!
lstm_4_373409:
!
lstm_5_373412:

lstm_5_373414:	!
lstm_5_373416:
,
time_distributed_2_373419:	9'
time_distributed_2_373421:9
identity˘lstm_4/StatefulPartitionedCall˘lstm_5/StatefulPartitionedCall˘*time_distributed_2/StatefulPartitionedCall
lstm_4/StatefulPartitionedCallStatefulPartitionedCalllstm_4_inputlstm_4_373405lstm_4_373407lstm_4_373409*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_373280¨
lstm_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0lstm_5_373412lstm_5_373414lstm_5_373416*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_372965Ć
*time_distributed_2/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0time_distributed_2_373419time_distributed_2_373421*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_372146q
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ź
time_distributed_2/ReshapeReshape'lstm_5/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
IdentityIdentity3time_distributed_2/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9ľ
NoOpNoOp^lstm_4/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall+^time_distributed_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : : : : : : 2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall2X
*time_distributed_2/StatefulPartitionedCall*time_distributed_2/StatefulPartitionedCall:b ^
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
&
_user_specified_namelstm_4_input
ńl
	
while_body_372261
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_4_split_readvariableop_resource_0:	9B
3while_lstm_cell_4_split_1_readvariableop_resource_0:	?
+while_lstm_cell_4_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_4_split_readvariableop_resource:	9@
1while_lstm_cell_4_split_1_readvariableop_resource:	=
)while_lstm_cell_4_readvariableop_resource:
˘ while/lstm_cell_4/ReadVariableOp˘"while/lstm_cell_4/ReadVariableOp_1˘"while/lstm_cell_4/ReadVariableOp_2˘"while/lstm_cell_4/ReadVariableOp_3˘&while/lstm_cell_4/split/ReadVariableOp˘(while/lstm_cell_4/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
element_dtype0d
!while/lstm_cell_4/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ž
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	9*
dtype0Ř
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_splitŠ
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Î
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mulMulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_1Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_2Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_3Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
while/lstm_cell_4/TanhTanhwhile/lstm_cell_4/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0while/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/lstm_cell_4/Tanh_1Tanhwhile/lstm_cell_4/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0while/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇy
while/Identity_4Identitywhile/lstm_cell_4/mul_6:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˛

while/NoOpNoOp!^while/lstm_cell_4/ReadVariableOp#^while/lstm_cell_4/ReadVariableOp_1#^while/lstm_cell_4/ReadVariableOp_2#^while/lstm_cell_4/ReadVariableOp_3'^while/lstm_cell_4/split/ReadVariableOp)^while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2D
 while/lstm_cell_4/ReadVariableOp while/lstm_cell_4/ReadVariableOp2H
"while/lstm_cell_4/ReadVariableOp_1"while/lstm_cell_4/ReadVariableOp_12H
"while/lstm_cell_4/ReadVariableOp_2"while/lstm_cell_4/ReadVariableOp_22H
"while/lstm_cell_4/ReadVariableOp_3"while/lstm_cell_4/ReadVariableOp_32P
&while/lstm_cell_4/split/ReadVariableOp&while/lstm_cell_4/split/ReadVariableOp2T
(while/lstm_cell_4/split_1/ReadVariableOp(while/lstm_cell_4/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 
˘?
Š
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_371245

inputs

states
states_10
split_readvariableop_resource:	9.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘split/ReadVariableOp˘split_1/ReadVariableOpE
ones_like/ShapeShapestates*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	9*
dtype0˘
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
mulMulstatesones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
mul_1Mulstatesones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
mul_2Mulstatesones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
mul_3Mulstatesones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskf
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙X
mul_4MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mul_5MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
IdentityIdentity	mul_6:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_1Identity	mul_6:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙9:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙9
 
_user_specified_nameinputs:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namestates:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namestates


Đ
-__inference_sequential_2_layer_call_fn_373377
lstm_4_input
unknown:	9
	unknown_0:	
	unknown_1:

	unknown_2:

	unknown_3:	
	unknown_4:

	unknown_5:	9
	unknown_6:9
identity˘StatefulPartitionedCallž
StatefulPartitionedCallStatefulPartitionedCalllstm_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_373337|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
&
_user_specified_namelstm_4_input
­
¸
'__inference_lstm_5_layer_call_fn_375684
inputs_0
unknown:

	unknown_0:	
	unknown_1:

identity˘StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_372062}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0
ĐĽ

lstm_4_while_body_374083*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3)
%lstm_4_while_lstm_4_strided_slice_1_0e
alstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0K
8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0:	9I
:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0:	F
2lstm_4_while_lstm_cell_4_readvariableop_resource_0:

lstm_4_while_identity
lstm_4_while_identity_1
lstm_4_while_identity_2
lstm_4_while_identity_3
lstm_4_while_identity_4
lstm_4_while_identity_5'
#lstm_4_while_lstm_4_strided_slice_1c
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensorI
6lstm_4_while_lstm_cell_4_split_readvariableop_resource:	9G
8lstm_4_while_lstm_cell_4_split_1_readvariableop_resource:	D
0lstm_4_while_lstm_cell_4_readvariableop_resource:
˘'lstm_4/while/lstm_cell_4/ReadVariableOp˘)lstm_4/while/lstm_cell_4/ReadVariableOp_1˘)lstm_4/while/lstm_cell_4/ReadVariableOp_2˘)lstm_4/while/lstm_cell_4/ReadVariableOp_3˘-lstm_4/while/lstm_cell_4/split/ReadVariableOp˘/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp
>lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   É
0lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0lstm_4_while_placeholderGlstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
element_dtype0r
(lstm_4/while/lstm_cell_4/ones_like/ShapeShapelstm_4_while_placeholder_2*
T0*
_output_shapes
:m
(lstm_4/while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ă
"lstm_4/while/lstm_cell_4/ones_likeFill1lstm_4/while/lstm_cell_4/ones_like/Shape:output:01lstm_4/while/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
&lstm_4/while/lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ź
$lstm_4/while/lstm_cell_4/dropout/MulMul+lstm_4/while/lstm_cell_4/ones_like:output:0/lstm_4/while/lstm_cell_4/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
&lstm_4/while/lstm_cell_4/dropout/ShapeShape+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:ż
=lstm_4/while/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform/lstm_4/while/lstm_cell_4/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0t
/lstm_4/while/lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ň
-lstm_4/while/lstm_cell_4/dropout/GreaterEqualGreaterEqualFlstm_4/while/lstm_cell_4/dropout/random_uniform/RandomUniform:output:08lstm_4/while/lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
%lstm_4/while/lstm_cell_4/dropout/CastCast1lstm_4/while/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ
&lstm_4/while/lstm_cell_4/dropout/Mul_1Mul(lstm_4/while/lstm_cell_4/dropout/Mul:z:0)lstm_4/while/lstm_cell_4/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
(lstm_4/while/lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ŕ
&lstm_4/while/lstm_cell_4/dropout_1/MulMul+lstm_4/while/lstm_cell_4/ones_like:output:01lstm_4/while/lstm_cell_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(lstm_4/while/lstm_cell_4/dropout_1/ShapeShape+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:Ă
?lstm_4/while/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform1lstm_4/while/lstm_cell_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0v
1lstm_4/while/lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ř
/lstm_4/while/lstm_cell_4/dropout_1/GreaterEqualGreaterEqualHlstm_4/while/lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:0:lstm_4/while/lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
'lstm_4/while/lstm_cell_4/dropout_1/CastCast3lstm_4/while/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ť
(lstm_4/while/lstm_cell_4/dropout_1/Mul_1Mul*lstm_4/while/lstm_cell_4/dropout_1/Mul:z:0+lstm_4/while/lstm_cell_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
(lstm_4/while/lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ŕ
&lstm_4/while/lstm_cell_4/dropout_2/MulMul+lstm_4/while/lstm_cell_4/ones_like:output:01lstm_4/while/lstm_cell_4/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(lstm_4/while/lstm_cell_4/dropout_2/ShapeShape+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:Ă
?lstm_4/while/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform1lstm_4/while/lstm_cell_4/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0v
1lstm_4/while/lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ř
/lstm_4/while/lstm_cell_4/dropout_2/GreaterEqualGreaterEqualHlstm_4/while/lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:0:lstm_4/while/lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
'lstm_4/while/lstm_cell_4/dropout_2/CastCast3lstm_4/while/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ť
(lstm_4/while/lstm_cell_4/dropout_2/Mul_1Mul*lstm_4/while/lstm_cell_4/dropout_2/Mul:z:0+lstm_4/while/lstm_cell_4/dropout_2/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
(lstm_4/while/lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ŕ
&lstm_4/while/lstm_cell_4/dropout_3/MulMul+lstm_4/while/lstm_cell_4/ones_like:output:01lstm_4/while/lstm_cell_4/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(lstm_4/while/lstm_cell_4/dropout_3/ShapeShape+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:Ă
?lstm_4/while/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform1lstm_4/while/lstm_cell_4/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0v
1lstm_4/while/lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ř
/lstm_4/while/lstm_cell_4/dropout_3/GreaterEqualGreaterEqualHlstm_4/while/lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:0:lstm_4/while/lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
'lstm_4/while/lstm_cell_4/dropout_3/CastCast3lstm_4/while/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ť
(lstm_4/while/lstm_cell_4/dropout_3/Mul_1Mul*lstm_4/while/lstm_cell_4/dropout_3/Mul:z:0+lstm_4/while/lstm_cell_4/dropout_3/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
(lstm_4/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :§
-lstm_4/while/lstm_cell_4/split/ReadVariableOpReadVariableOp8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	9*
dtype0í
lstm_4/while/lstm_cell_4/splitSplit1lstm_4/while/lstm_cell_4/split/split_dim:output:05lstm_4/while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_splitž
lstm_4/while/lstm_cell_4/MatMulMatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
!lstm_4/while/lstm_cell_4/MatMul_1MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
!lstm_4/while/lstm_cell_4/MatMul_2MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
!lstm_4/while/lstm_cell_4/MatMul_3MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l
*lstm_4/while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : §
/lstm_4/while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0ă
 lstm_4/while/lstm_cell_4/split_1Split3lstm_4/while/lstm_cell_4/split_1/split_dim:output:07lstm_4/while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split´
 lstm_4/while/lstm_cell_4/BiasAddBiasAdd)lstm_4/while/lstm_cell_4/MatMul:product:0)lstm_4/while/lstm_cell_4/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
"lstm_4/while/lstm_cell_4/BiasAdd_1BiasAdd+lstm_4/while/lstm_cell_4/MatMul_1:product:0)lstm_4/while/lstm_cell_4/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
"lstm_4/while/lstm_cell_4/BiasAdd_2BiasAdd+lstm_4/while/lstm_cell_4/MatMul_2:product:0)lstm_4/while/lstm_cell_4/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
"lstm_4/while/lstm_cell_4/BiasAdd_3BiasAdd+lstm_4/while/lstm_cell_4/MatMul_3:product:0)lstm_4/while/lstm_cell_4/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/while/lstm_cell_4/mulMullstm_4_while_placeholder_2*lstm_4/while/lstm_cell_4/dropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_4/while/lstm_cell_4/mul_1Mullstm_4_while_placeholder_2,lstm_4/while/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_4/while/lstm_cell_4/mul_2Mullstm_4_while_placeholder_2,lstm_4/while/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_4/while/lstm_cell_4/mul_3Mullstm_4_while_placeholder_2,lstm_4/while/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
'lstm_4/while/lstm_cell_4/ReadVariableOpReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0}
,lstm_4/while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.lstm_4/while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.lstm_4/while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ę
&lstm_4/while/lstm_cell_4/strided_sliceStridedSlice/lstm_4/while/lstm_cell_4/ReadVariableOp:value:05lstm_4/while/lstm_cell_4/strided_slice/stack:output:07lstm_4/while/lstm_cell_4/strided_slice/stack_1:output:07lstm_4/while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maską
!lstm_4/while/lstm_cell_4/MatMul_4MatMul lstm_4/while/lstm_cell_4/mul:z:0/lstm_4/while/lstm_cell_4/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙°
lstm_4/while/lstm_cell_4/addAddV2)lstm_4/while/lstm_cell_4/BiasAdd:output:0+lstm_4/while/lstm_cell_4/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 lstm_4/while/lstm_cell_4/SigmoidSigmoid lstm_4/while/lstm_cell_4/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)lstm_4/while/lstm_cell_4/ReadVariableOp_1ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
.lstm_4/while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
0lstm_4/while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
0lstm_4/while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(lstm_4/while/lstm_cell_4/strided_slice_1StridedSlice1lstm_4/while/lstm_cell_4/ReadVariableOp_1:value:07lstm_4/while/lstm_cell_4/strided_slice_1/stack:output:09lstm_4/while/lstm_cell_4/strided_slice_1/stack_1:output:09lstm_4/while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskľ
!lstm_4/while/lstm_cell_4/MatMul_5MatMul"lstm_4/while/lstm_cell_4/mul_1:z:01lstm_4/while/lstm_cell_4/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙´
lstm_4/while/lstm_cell_4/add_1AddV2+lstm_4/while/lstm_cell_4/BiasAdd_1:output:0+lstm_4/while/lstm_cell_4/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"lstm_4/while/lstm_cell_4/Sigmoid_1Sigmoid"lstm_4/while/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/while/lstm_cell_4/mul_4Mul&lstm_4/while/lstm_cell_4/Sigmoid_1:y:0lstm_4_while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)lstm_4/while/lstm_cell_4/ReadVariableOp_2ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
.lstm_4/while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
0lstm_4/while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
0lstm_4/while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(lstm_4/while/lstm_cell_4/strided_slice_2StridedSlice1lstm_4/while/lstm_cell_4/ReadVariableOp_2:value:07lstm_4/while/lstm_cell_4/strided_slice_2/stack:output:09lstm_4/while/lstm_cell_4/strided_slice_2/stack_1:output:09lstm_4/while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskľ
!lstm_4/while/lstm_cell_4/MatMul_6MatMul"lstm_4/while/lstm_cell_4/mul_2:z:01lstm_4/while/lstm_cell_4/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙´
lstm_4/while/lstm_cell_4/add_2AddV2+lstm_4/while/lstm_cell_4/BiasAdd_2:output:0+lstm_4/while/lstm_cell_4/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_4/while/lstm_cell_4/TanhTanh"lstm_4/while/lstm_cell_4/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
lstm_4/while/lstm_cell_4/mul_5Mul$lstm_4/while/lstm_cell_4/Sigmoid:y:0!lstm_4/while/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_4/while/lstm_cell_4/add_3AddV2"lstm_4/while/lstm_cell_4/mul_4:z:0"lstm_4/while/lstm_cell_4/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)lstm_4/while/lstm_cell_4/ReadVariableOp_3ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
.lstm_4/while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
0lstm_4/while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0lstm_4/while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(lstm_4/while/lstm_cell_4/strided_slice_3StridedSlice1lstm_4/while/lstm_cell_4/ReadVariableOp_3:value:07lstm_4/while/lstm_cell_4/strided_slice_3/stack:output:09lstm_4/while/lstm_cell_4/strided_slice_3/stack_1:output:09lstm_4/while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskľ
!lstm_4/while/lstm_cell_4/MatMul_7MatMul"lstm_4/while/lstm_cell_4/mul_3:z:01lstm_4/while/lstm_cell_4/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙´
lstm_4/while/lstm_cell_4/add_4AddV2+lstm_4/while/lstm_cell_4/BiasAdd_3:output:0+lstm_4/while/lstm_cell_4/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"lstm_4/while/lstm_cell_4/Sigmoid_2Sigmoid"lstm_4/while/lstm_cell_4/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_4/while/lstm_cell_4/Tanh_1Tanh"lstm_4/while/lstm_cell_4/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ľ
lstm_4/while/lstm_cell_4/mul_6Mul&lstm_4/while/lstm_cell_4/Sigmoid_2:y:0#lstm_4/while/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ŕ
1lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_4_while_placeholder_1lstm_4_while_placeholder"lstm_4/while/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype0:éčŇT
lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_4/while/addAddV2lstm_4_while_placeholderlstm_4/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_4/while/add_1AddV2&lstm_4_while_lstm_4_while_loop_counterlstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_4/while/IdentityIdentitylstm_4/while/add_1:z:0^lstm_4/while/NoOp*
T0*
_output_shapes
: 
lstm_4/while/Identity_1Identity,lstm_4_while_lstm_4_while_maximum_iterations^lstm_4/while/NoOp*
T0*
_output_shapes
: n
lstm_4/while/Identity_2Identitylstm_4/while/add:z:0^lstm_4/while/NoOp*
T0*
_output_shapes
: Ž
lstm_4/while/Identity_3IdentityAlstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_4/while/NoOp*
T0*
_output_shapes
: :éčŇ
lstm_4/while/Identity_4Identity"lstm_4/while/lstm_cell_4/mul_6:z:0^lstm_4/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/while/Identity_5Identity"lstm_4/while/lstm_cell_4/add_3:z:0^lstm_4/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ă
lstm_4/while/NoOpNoOp(^lstm_4/while/lstm_cell_4/ReadVariableOp*^lstm_4/while/lstm_cell_4/ReadVariableOp_1*^lstm_4/while/lstm_cell_4/ReadVariableOp_2*^lstm_4/while/lstm_cell_4/ReadVariableOp_3.^lstm_4/while/lstm_cell_4/split/ReadVariableOp0^lstm_4/while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_4_while_identitylstm_4/while/Identity:output:0";
lstm_4_while_identity_1 lstm_4/while/Identity_1:output:0";
lstm_4_while_identity_2 lstm_4/while/Identity_2:output:0";
lstm_4_while_identity_3 lstm_4/while/Identity_3:output:0";
lstm_4_while_identity_4 lstm_4/while/Identity_4:output:0";
lstm_4_while_identity_5 lstm_4/while/Identity_5:output:0"L
#lstm_4_while_lstm_4_strided_slice_1%lstm_4_while_lstm_4_strided_slice_1_0"f
0lstm_4_while_lstm_cell_4_readvariableop_resource2lstm_4_while_lstm_cell_4_readvariableop_resource_0"v
8lstm_4_while_lstm_cell_4_split_1_readvariableop_resource:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0"r
6lstm_4_while_lstm_cell_4_split_readvariableop_resource8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0"Ä
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensoralstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2R
'lstm_4/while/lstm_cell_4/ReadVariableOp'lstm_4/while/lstm_cell_4/ReadVariableOp2V
)lstm_4/while/lstm_cell_4/ReadVariableOp_1)lstm_4/while/lstm_cell_4/ReadVariableOp_12V
)lstm_4/while/lstm_cell_4/ReadVariableOp_2)lstm_4/while/lstm_cell_4/ReadVariableOp_22V
)lstm_4/while/lstm_cell_4/ReadVariableOp_3)lstm_4/while/lstm_cell_4/ReadVariableOp_32^
-lstm_4/while/lstm_cell_4/split/ReadVariableOp-lstm_4/while/lstm_cell_4/split/ReadVariableOp2b
/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 
§
Ź
"__inference__traced_restore_377483
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: ?
,assignvariableop_5_lstm_4_lstm_cell_4_kernel:	9J
6assignvariableop_6_lstm_4_lstm_cell_4_recurrent_kernel:
9
*assignvariableop_7_lstm_4_lstm_cell_4_bias:	@
,assignvariableop_8_lstm_5_lstm_cell_5_kernel:
J
6assignvariableop_9_lstm_5_lstm_cell_5_recurrent_kernel:
:
+assignvariableop_10_lstm_5_lstm_cell_5_bias:	@
-assignvariableop_11_time_distributed_2_kernel:	99
+assignvariableop_12_time_distributed_2_bias:9#
assignvariableop_13_total: #
assignvariableop_14_count: G
4assignvariableop_15_adam_lstm_4_lstm_cell_4_kernel_m:	9R
>assignvariableop_16_adam_lstm_4_lstm_cell_4_recurrent_kernel_m:
A
2assignvariableop_17_adam_lstm_4_lstm_cell_4_bias_m:	H
4assignvariableop_18_adam_lstm_5_lstm_cell_5_kernel_m:
R
>assignvariableop_19_adam_lstm_5_lstm_cell_5_recurrent_kernel_m:
A
2assignvariableop_20_adam_lstm_5_lstm_cell_5_bias_m:	G
4assignvariableop_21_adam_time_distributed_2_kernel_m:	9@
2assignvariableop_22_adam_time_distributed_2_bias_m:9G
4assignvariableop_23_adam_lstm_4_lstm_cell_4_kernel_v:	9R
>assignvariableop_24_adam_lstm_4_lstm_cell_4_recurrent_kernel_v:
A
2assignvariableop_25_adam_lstm_4_lstm_cell_4_bias_v:	H
4assignvariableop_26_adam_lstm_5_lstm_cell_5_kernel_v:
R
>assignvariableop_27_adam_lstm_5_lstm_cell_5_recurrent_kernel_v:
A
2assignvariableop_28_adam_lstm_5_lstm_cell_5_bias_v:	G
4assignvariableop_29_adam_time_distributed_2_kernel_v:	9@
2assignvariableop_30_adam_time_distributed_2_bias_v:9
identity_32˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_18˘AssignVariableOp_19˘AssignVariableOp_2˘AssignVariableOp_20˘AssignVariableOp_21˘AssignVariableOp_22˘AssignVariableOp_23˘AssignVariableOp_24˘AssignVariableOp_25˘AssignVariableOp_26˘AssignVariableOp_27˘AssignVariableOp_28˘AssignVariableOp_29˘AssignVariableOp_3˘AssignVariableOp_30˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9Ţ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueúB÷ B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH°
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Á
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp,assignvariableop_5_lstm_4_lstm_cell_4_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ľ
AssignVariableOp_6AssignVariableOp6assignvariableop_6_lstm_4_lstm_cell_4_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp*assignvariableop_7_lstm_4_lstm_cell_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp,assignvariableop_8_lstm_5_lstm_cell_5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ľ
AssignVariableOp_9AssignVariableOp6assignvariableop_9_lstm_5_lstm_cell_5_recurrent_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp+assignvariableop_10_lstm_5_lstm_cell_5_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp-assignvariableop_11_time_distributed_2_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp+assignvariableop_12_time_distributed_2_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ľ
AssignVariableOp_15AssignVariableOp4assignvariableop_15_adam_lstm_4_lstm_cell_4_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ż
AssignVariableOp_16AssignVariableOp>assignvariableop_16_adam_lstm_4_lstm_cell_4_recurrent_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ł
AssignVariableOp_17AssignVariableOp2assignvariableop_17_adam_lstm_4_lstm_cell_4_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ľ
AssignVariableOp_18AssignVariableOp4assignvariableop_18_adam_lstm_5_lstm_cell_5_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ż
AssignVariableOp_19AssignVariableOp>assignvariableop_19_adam_lstm_5_lstm_cell_5_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ł
AssignVariableOp_20AssignVariableOp2assignvariableop_20_adam_lstm_5_lstm_cell_5_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ľ
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_time_distributed_2_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ł
AssignVariableOp_22AssignVariableOp2assignvariableop_22_adam_time_distributed_2_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ľ
AssignVariableOp_23AssignVariableOp4assignvariableop_23_adam_lstm_4_lstm_cell_4_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ż
AssignVariableOp_24AssignVariableOp>assignvariableop_24_adam_lstm_4_lstm_cell_4_recurrent_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ł
AssignVariableOp_25AssignVariableOp2assignvariableop_25_adam_lstm_4_lstm_cell_4_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ľ
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_lstm_5_lstm_cell_5_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ż
AssignVariableOp_27AssignVariableOp>assignvariableop_27_adam_lstm_5_lstm_cell_5_recurrent_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ł
AssignVariableOp_28AssignVariableOp2assignvariableop_28_adam_lstm_5_lstm_cell_5_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ľ
AssignVariableOp_29AssignVariableOp4assignvariableop_29_adam_time_distributed_2_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ł
AssignVariableOp_30AssignVariableOp2assignvariableop_30_adam_time_distributed_2_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ů
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: ć
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ŘĽ

lstm_5_while_body_374372*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3)
%lstm_5_while_lstm_5_strided_slice_1_0e
alstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0L
8lstm_5_while_lstm_cell_5_split_readvariableop_resource_0:
I
:lstm_5_while_lstm_cell_5_split_1_readvariableop_resource_0:	F
2lstm_5_while_lstm_cell_5_readvariableop_resource_0:

lstm_5_while_identity
lstm_5_while_identity_1
lstm_5_while_identity_2
lstm_5_while_identity_3
lstm_5_while_identity_4
lstm_5_while_identity_5'
#lstm_5_while_lstm_5_strided_slice_1c
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensorJ
6lstm_5_while_lstm_cell_5_split_readvariableop_resource:
G
8lstm_5_while_lstm_cell_5_split_1_readvariableop_resource:	D
0lstm_5_while_lstm_cell_5_readvariableop_resource:
˘'lstm_5/while/lstm_cell_5/ReadVariableOp˘)lstm_5/while/lstm_cell_5/ReadVariableOp_1˘)lstm_5/while/lstm_cell_5/ReadVariableOp_2˘)lstm_5/while/lstm_cell_5/ReadVariableOp_3˘-lstm_5/while/lstm_cell_5/split/ReadVariableOp˘/lstm_5/while/lstm_cell_5/split_1/ReadVariableOp
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ę
0lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0lstm_5_while_placeholderGlstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0r
(lstm_5/while/lstm_cell_5/ones_like/ShapeShapelstm_5_while_placeholder_2*
T0*
_output_shapes
:m
(lstm_5/while/lstm_cell_5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ă
"lstm_5/while/lstm_cell_5/ones_likeFill1lstm_5/while/lstm_cell_5/ones_like/Shape:output:01lstm_5/while/lstm_cell_5/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
&lstm_5/while/lstm_cell_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ź
$lstm_5/while/lstm_cell_5/dropout/MulMul+lstm_5/while/lstm_cell_5/ones_like:output:0/lstm_5/while/lstm_cell_5/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
&lstm_5/while/lstm_cell_5/dropout/ShapeShape+lstm_5/while/lstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:ż
=lstm_5/while/lstm_cell_5/dropout/random_uniform/RandomUniformRandomUniform/lstm_5/while/lstm_cell_5/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0t
/lstm_5/while/lstm_cell_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ň
-lstm_5/while/lstm_cell_5/dropout/GreaterEqualGreaterEqualFlstm_5/while/lstm_cell_5/dropout/random_uniform/RandomUniform:output:08lstm_5/while/lstm_cell_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
%lstm_5/while/lstm_cell_5/dropout/CastCast1lstm_5/while/lstm_cell_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ
&lstm_5/while/lstm_cell_5/dropout/Mul_1Mul(lstm_5/while/lstm_cell_5/dropout/Mul:z:0)lstm_5/while/lstm_cell_5/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
(lstm_5/while/lstm_cell_5/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ŕ
&lstm_5/while/lstm_cell_5/dropout_1/MulMul+lstm_5/while/lstm_cell_5/ones_like:output:01lstm_5/while/lstm_cell_5/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(lstm_5/while/lstm_cell_5/dropout_1/ShapeShape+lstm_5/while/lstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:Ă
?lstm_5/while/lstm_cell_5/dropout_1/random_uniform/RandomUniformRandomUniform1lstm_5/while/lstm_cell_5/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0v
1lstm_5/while/lstm_cell_5/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ř
/lstm_5/while/lstm_cell_5/dropout_1/GreaterEqualGreaterEqualHlstm_5/while/lstm_cell_5/dropout_1/random_uniform/RandomUniform:output:0:lstm_5/while/lstm_cell_5/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
'lstm_5/while/lstm_cell_5/dropout_1/CastCast3lstm_5/while/lstm_cell_5/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ť
(lstm_5/while/lstm_cell_5/dropout_1/Mul_1Mul*lstm_5/while/lstm_cell_5/dropout_1/Mul:z:0+lstm_5/while/lstm_cell_5/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
(lstm_5/while/lstm_cell_5/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ŕ
&lstm_5/while/lstm_cell_5/dropout_2/MulMul+lstm_5/while/lstm_cell_5/ones_like:output:01lstm_5/while/lstm_cell_5/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(lstm_5/while/lstm_cell_5/dropout_2/ShapeShape+lstm_5/while/lstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:Ă
?lstm_5/while/lstm_cell_5/dropout_2/random_uniform/RandomUniformRandomUniform1lstm_5/while/lstm_cell_5/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0v
1lstm_5/while/lstm_cell_5/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ř
/lstm_5/while/lstm_cell_5/dropout_2/GreaterEqualGreaterEqualHlstm_5/while/lstm_cell_5/dropout_2/random_uniform/RandomUniform:output:0:lstm_5/while/lstm_cell_5/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
'lstm_5/while/lstm_cell_5/dropout_2/CastCast3lstm_5/while/lstm_cell_5/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ť
(lstm_5/while/lstm_cell_5/dropout_2/Mul_1Mul*lstm_5/while/lstm_cell_5/dropout_2/Mul:z:0+lstm_5/while/lstm_cell_5/dropout_2/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
(lstm_5/while/lstm_cell_5/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ŕ
&lstm_5/while/lstm_cell_5/dropout_3/MulMul+lstm_5/while/lstm_cell_5/ones_like:output:01lstm_5/while/lstm_cell_5/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(lstm_5/while/lstm_cell_5/dropout_3/ShapeShape+lstm_5/while/lstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:Ă
?lstm_5/while/lstm_cell_5/dropout_3/random_uniform/RandomUniformRandomUniform1lstm_5/while/lstm_cell_5/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0v
1lstm_5/while/lstm_cell_5/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ř
/lstm_5/while/lstm_cell_5/dropout_3/GreaterEqualGreaterEqualHlstm_5/while/lstm_cell_5/dropout_3/random_uniform/RandomUniform:output:0:lstm_5/while/lstm_cell_5/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
'lstm_5/while/lstm_cell_5/dropout_3/CastCast3lstm_5/while/lstm_cell_5/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ť
(lstm_5/while/lstm_cell_5/dropout_3/Mul_1Mul*lstm_5/while/lstm_cell_5/dropout_3/Mul:z:0+lstm_5/while/lstm_cell_5/dropout_3/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
(lstm_5/while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¨
-lstm_5/while/lstm_cell_5/split/ReadVariableOpReadVariableOp8lstm_5_while_lstm_cell_5_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype0ń
lstm_5/while/lstm_cell_5/splitSplit1lstm_5/while/lstm_cell_5/split/split_dim:output:05lstm_5/while/lstm_cell_5/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_splitž
lstm_5/while/lstm_cell_5/MatMulMatMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_5/while/lstm_cell_5/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
!lstm_5/while/lstm_cell_5/MatMul_1MatMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_5/while/lstm_cell_5/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
!lstm_5/while/lstm_cell_5/MatMul_2MatMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_5/while/lstm_cell_5/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
!lstm_5/while/lstm_cell_5/MatMul_3MatMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_5/while/lstm_cell_5/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l
*lstm_5/while/lstm_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : §
/lstm_5/while/lstm_cell_5/split_1/ReadVariableOpReadVariableOp:lstm_5_while_lstm_cell_5_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0ă
 lstm_5/while/lstm_cell_5/split_1Split3lstm_5/while/lstm_cell_5/split_1/split_dim:output:07lstm_5/while/lstm_cell_5/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split´
 lstm_5/while/lstm_cell_5/BiasAddBiasAdd)lstm_5/while/lstm_cell_5/MatMul:product:0)lstm_5/while/lstm_cell_5/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
"lstm_5/while/lstm_cell_5/BiasAdd_1BiasAdd+lstm_5/while/lstm_cell_5/MatMul_1:product:0)lstm_5/while/lstm_cell_5/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
"lstm_5/while/lstm_cell_5/BiasAdd_2BiasAdd+lstm_5/while/lstm_cell_5/MatMul_2:product:0)lstm_5/while/lstm_cell_5/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
"lstm_5/while/lstm_cell_5/BiasAdd_3BiasAdd+lstm_5/while/lstm_cell_5/MatMul_3:product:0)lstm_5/while/lstm_cell_5/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/while/lstm_cell_5/mulMullstm_5_while_placeholder_2*lstm_5/while/lstm_cell_5/dropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_5/while/lstm_cell_5/mul_1Mullstm_5_while_placeholder_2,lstm_5/while/lstm_cell_5/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_5/while/lstm_cell_5/mul_2Mullstm_5_while_placeholder_2,lstm_5/while/lstm_cell_5/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_5/while/lstm_cell_5/mul_3Mullstm_5_while_placeholder_2,lstm_5/while/lstm_cell_5/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
'lstm_5/while/lstm_cell_5/ReadVariableOpReadVariableOp2lstm_5_while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0}
,lstm_5/while/lstm_cell_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.lstm_5/while/lstm_cell_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.lstm_5/while/lstm_cell_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ę
&lstm_5/while/lstm_cell_5/strided_sliceStridedSlice/lstm_5/while/lstm_cell_5/ReadVariableOp:value:05lstm_5/while/lstm_cell_5/strided_slice/stack:output:07lstm_5/while/lstm_cell_5/strided_slice/stack_1:output:07lstm_5/while/lstm_cell_5/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maską
!lstm_5/while/lstm_cell_5/MatMul_4MatMul lstm_5/while/lstm_cell_5/mul:z:0/lstm_5/while/lstm_cell_5/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙°
lstm_5/while/lstm_cell_5/addAddV2)lstm_5/while/lstm_cell_5/BiasAdd:output:0+lstm_5/while/lstm_cell_5/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 lstm_5/while/lstm_cell_5/SigmoidSigmoid lstm_5/while/lstm_cell_5/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)lstm_5/while/lstm_cell_5/ReadVariableOp_1ReadVariableOp2lstm_5_while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
.lstm_5/while/lstm_cell_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
0lstm_5/while/lstm_cell_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
0lstm_5/while/lstm_cell_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(lstm_5/while/lstm_cell_5/strided_slice_1StridedSlice1lstm_5/while/lstm_cell_5/ReadVariableOp_1:value:07lstm_5/while/lstm_cell_5/strided_slice_1/stack:output:09lstm_5/while/lstm_cell_5/strided_slice_1/stack_1:output:09lstm_5/while/lstm_cell_5/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskľ
!lstm_5/while/lstm_cell_5/MatMul_5MatMul"lstm_5/while/lstm_cell_5/mul_1:z:01lstm_5/while/lstm_cell_5/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙´
lstm_5/while/lstm_cell_5/add_1AddV2+lstm_5/while/lstm_cell_5/BiasAdd_1:output:0+lstm_5/while/lstm_cell_5/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"lstm_5/while/lstm_cell_5/Sigmoid_1Sigmoid"lstm_5/while/lstm_cell_5/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/while/lstm_cell_5/mul_4Mul&lstm_5/while/lstm_cell_5/Sigmoid_1:y:0lstm_5_while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)lstm_5/while/lstm_cell_5/ReadVariableOp_2ReadVariableOp2lstm_5_while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
.lstm_5/while/lstm_cell_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
0lstm_5/while/lstm_cell_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
0lstm_5/while/lstm_cell_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(lstm_5/while/lstm_cell_5/strided_slice_2StridedSlice1lstm_5/while/lstm_cell_5/ReadVariableOp_2:value:07lstm_5/while/lstm_cell_5/strided_slice_2/stack:output:09lstm_5/while/lstm_cell_5/strided_slice_2/stack_1:output:09lstm_5/while/lstm_cell_5/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskľ
!lstm_5/while/lstm_cell_5/MatMul_6MatMul"lstm_5/while/lstm_cell_5/mul_2:z:01lstm_5/while/lstm_cell_5/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙´
lstm_5/while/lstm_cell_5/add_2AddV2+lstm_5/while/lstm_cell_5/BiasAdd_2:output:0+lstm_5/while/lstm_cell_5/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_5/while/lstm_cell_5/TanhTanh"lstm_5/while/lstm_cell_5/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
lstm_5/while/lstm_cell_5/mul_5Mul$lstm_5/while/lstm_cell_5/Sigmoid:y:0!lstm_5/while/lstm_cell_5/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_5/while/lstm_cell_5/add_3AddV2"lstm_5/while/lstm_cell_5/mul_4:z:0"lstm_5/while/lstm_cell_5/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)lstm_5/while/lstm_cell_5/ReadVariableOp_3ReadVariableOp2lstm_5_while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
.lstm_5/while/lstm_cell_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
0lstm_5/while/lstm_cell_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0lstm_5/while/lstm_cell_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(lstm_5/while/lstm_cell_5/strided_slice_3StridedSlice1lstm_5/while/lstm_cell_5/ReadVariableOp_3:value:07lstm_5/while/lstm_cell_5/strided_slice_3/stack:output:09lstm_5/while/lstm_cell_5/strided_slice_3/stack_1:output:09lstm_5/while/lstm_cell_5/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskľ
!lstm_5/while/lstm_cell_5/MatMul_7MatMul"lstm_5/while/lstm_cell_5/mul_3:z:01lstm_5/while/lstm_cell_5/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙´
lstm_5/while/lstm_cell_5/add_4AddV2+lstm_5/while/lstm_cell_5/BiasAdd_3:output:0+lstm_5/while/lstm_cell_5/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"lstm_5/while/lstm_cell_5/Sigmoid_2Sigmoid"lstm_5/while/lstm_cell_5/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_5/while/lstm_cell_5/Tanh_1Tanh"lstm_5/while/lstm_cell_5/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ľ
lstm_5/while/lstm_cell_5/mul_6Mul&lstm_5/while/lstm_cell_5/Sigmoid_2:y:0#lstm_5/while/lstm_cell_5/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ŕ
1lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_5_while_placeholder_1lstm_5_while_placeholder"lstm_5/while/lstm_cell_5/mul_6:z:0*
_output_shapes
: *
element_dtype0:éčŇT
lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_5/while/addAddV2lstm_5_while_placeholderlstm_5/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_5/while/add_1AddV2&lstm_5_while_lstm_5_while_loop_counterlstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_5/while/IdentityIdentitylstm_5/while/add_1:z:0^lstm_5/while/NoOp*
T0*
_output_shapes
: 
lstm_5/while/Identity_1Identity,lstm_5_while_lstm_5_while_maximum_iterations^lstm_5/while/NoOp*
T0*
_output_shapes
: n
lstm_5/while/Identity_2Identitylstm_5/while/add:z:0^lstm_5/while/NoOp*
T0*
_output_shapes
: Ž
lstm_5/while/Identity_3IdentityAlstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_5/while/NoOp*
T0*
_output_shapes
: :éčŇ
lstm_5/while/Identity_4Identity"lstm_5/while/lstm_cell_5/mul_6:z:0^lstm_5/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/while/Identity_5Identity"lstm_5/while/lstm_cell_5/add_3:z:0^lstm_5/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ă
lstm_5/while/NoOpNoOp(^lstm_5/while/lstm_cell_5/ReadVariableOp*^lstm_5/while/lstm_cell_5/ReadVariableOp_1*^lstm_5/while/lstm_cell_5/ReadVariableOp_2*^lstm_5/while/lstm_cell_5/ReadVariableOp_3.^lstm_5/while/lstm_cell_5/split/ReadVariableOp0^lstm_5/while/lstm_cell_5/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_5_while_identitylstm_5/while/Identity:output:0";
lstm_5_while_identity_1 lstm_5/while/Identity_1:output:0";
lstm_5_while_identity_2 lstm_5/while/Identity_2:output:0";
lstm_5_while_identity_3 lstm_5/while/Identity_3:output:0";
lstm_5_while_identity_4 lstm_5/while/Identity_4:output:0";
lstm_5_while_identity_5 lstm_5/while/Identity_5:output:0"L
#lstm_5_while_lstm_5_strided_slice_1%lstm_5_while_lstm_5_strided_slice_1_0"f
0lstm_5_while_lstm_cell_5_readvariableop_resource2lstm_5_while_lstm_cell_5_readvariableop_resource_0"v
8lstm_5_while_lstm_cell_5_split_1_readvariableop_resource:lstm_5_while_lstm_cell_5_split_1_readvariableop_resource_0"r
6lstm_5_while_lstm_cell_5_split_readvariableop_resource8lstm_5_while_lstm_cell_5_split_readvariableop_resource_0"Ä
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensoralstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2R
'lstm_5/while/lstm_cell_5/ReadVariableOp'lstm_5/while/lstm_cell_5/ReadVariableOp2V
)lstm_5/while/lstm_cell_5/ReadVariableOp_1)lstm_5/while/lstm_cell_5/ReadVariableOp_12V
)lstm_5/while/lstm_cell_5/ReadVariableOp_2)lstm_5/while/lstm_cell_5/ReadVariableOp_22V
)lstm_5/while/lstm_cell_5/ReadVariableOp_3)lstm_5/while/lstm_cell_5/ReadVariableOp_32^
-lstm_5/while/lstm_cell_5/split/ReadVariableOp-lstm_5/while/lstm_cell_5/split/ReadVariableOp2b
/lstm_5/while/lstm_cell_5/split_1/ReadVariableOp/lstm_5/while/lstm_cell_5/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 

Ą
3__inference_time_distributed_2_layer_call_fn_376768

inputs
unknown:	9
	unknown_0:9
identity˘StatefulPartitionedCallđ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_372146|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
{

lstm_5_while_body_373802*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3)
%lstm_5_while_lstm_5_strided_slice_1_0e
alstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0L
8lstm_5_while_lstm_cell_5_split_readvariableop_resource_0:
I
:lstm_5_while_lstm_cell_5_split_1_readvariableop_resource_0:	F
2lstm_5_while_lstm_cell_5_readvariableop_resource_0:

lstm_5_while_identity
lstm_5_while_identity_1
lstm_5_while_identity_2
lstm_5_while_identity_3
lstm_5_while_identity_4
lstm_5_while_identity_5'
#lstm_5_while_lstm_5_strided_slice_1c
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensorJ
6lstm_5_while_lstm_cell_5_split_readvariableop_resource:
G
8lstm_5_while_lstm_cell_5_split_1_readvariableop_resource:	D
0lstm_5_while_lstm_cell_5_readvariableop_resource:
˘'lstm_5/while/lstm_cell_5/ReadVariableOp˘)lstm_5/while/lstm_cell_5/ReadVariableOp_1˘)lstm_5/while/lstm_cell_5/ReadVariableOp_2˘)lstm_5/while/lstm_cell_5/ReadVariableOp_3˘-lstm_5/while/lstm_cell_5/split/ReadVariableOp˘/lstm_5/while/lstm_cell_5/split_1/ReadVariableOp
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ę
0lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0lstm_5_while_placeholderGlstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0r
(lstm_5/while/lstm_cell_5/ones_like/ShapeShapelstm_5_while_placeholder_2*
T0*
_output_shapes
:m
(lstm_5/while/lstm_cell_5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ă
"lstm_5/while/lstm_cell_5/ones_likeFill1lstm_5/while/lstm_cell_5/ones_like/Shape:output:01lstm_5/while/lstm_cell_5/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
(lstm_5/while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¨
-lstm_5/while/lstm_cell_5/split/ReadVariableOpReadVariableOp8lstm_5_while_lstm_cell_5_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype0ń
lstm_5/while/lstm_cell_5/splitSplit1lstm_5/while/lstm_cell_5/split/split_dim:output:05lstm_5/while/lstm_cell_5/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_splitž
lstm_5/while/lstm_cell_5/MatMulMatMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_5/while/lstm_cell_5/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
!lstm_5/while/lstm_cell_5/MatMul_1MatMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_5/while/lstm_cell_5/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
!lstm_5/while/lstm_cell_5/MatMul_2MatMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_5/while/lstm_cell_5/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
!lstm_5/while/lstm_cell_5/MatMul_3MatMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_5/while/lstm_cell_5/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l
*lstm_5/while/lstm_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : §
/lstm_5/while/lstm_cell_5/split_1/ReadVariableOpReadVariableOp:lstm_5_while_lstm_cell_5_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0ă
 lstm_5/while/lstm_cell_5/split_1Split3lstm_5/while/lstm_cell_5/split_1/split_dim:output:07lstm_5/while/lstm_cell_5/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split´
 lstm_5/while/lstm_cell_5/BiasAddBiasAdd)lstm_5/while/lstm_cell_5/MatMul:product:0)lstm_5/while/lstm_cell_5/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
"lstm_5/while/lstm_cell_5/BiasAdd_1BiasAdd+lstm_5/while/lstm_cell_5/MatMul_1:product:0)lstm_5/while/lstm_cell_5/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
"lstm_5/while/lstm_cell_5/BiasAdd_2BiasAdd+lstm_5/while/lstm_cell_5/MatMul_2:product:0)lstm_5/while/lstm_cell_5/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
"lstm_5/while/lstm_cell_5/BiasAdd_3BiasAdd+lstm_5/while/lstm_cell_5/MatMul_3:product:0)lstm_5/while/lstm_cell_5/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/while/lstm_cell_5/mulMullstm_5_while_placeholder_2+lstm_5/while/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
lstm_5/while/lstm_cell_5/mul_1Mullstm_5_while_placeholder_2+lstm_5/while/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
lstm_5/while/lstm_cell_5/mul_2Mullstm_5_while_placeholder_2+lstm_5/while/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
lstm_5/while/lstm_cell_5/mul_3Mullstm_5_while_placeholder_2+lstm_5/while/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
'lstm_5/while/lstm_cell_5/ReadVariableOpReadVariableOp2lstm_5_while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0}
,lstm_5/while/lstm_cell_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.lstm_5/while/lstm_cell_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.lstm_5/while/lstm_cell_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ę
&lstm_5/while/lstm_cell_5/strided_sliceStridedSlice/lstm_5/while/lstm_cell_5/ReadVariableOp:value:05lstm_5/while/lstm_cell_5/strided_slice/stack:output:07lstm_5/while/lstm_cell_5/strided_slice/stack_1:output:07lstm_5/while/lstm_cell_5/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maską
!lstm_5/while/lstm_cell_5/MatMul_4MatMul lstm_5/while/lstm_cell_5/mul:z:0/lstm_5/while/lstm_cell_5/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙°
lstm_5/while/lstm_cell_5/addAddV2)lstm_5/while/lstm_cell_5/BiasAdd:output:0+lstm_5/while/lstm_cell_5/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 lstm_5/while/lstm_cell_5/SigmoidSigmoid lstm_5/while/lstm_cell_5/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)lstm_5/while/lstm_cell_5/ReadVariableOp_1ReadVariableOp2lstm_5_while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
.lstm_5/while/lstm_cell_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
0lstm_5/while/lstm_cell_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
0lstm_5/while/lstm_cell_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(lstm_5/while/lstm_cell_5/strided_slice_1StridedSlice1lstm_5/while/lstm_cell_5/ReadVariableOp_1:value:07lstm_5/while/lstm_cell_5/strided_slice_1/stack:output:09lstm_5/while/lstm_cell_5/strided_slice_1/stack_1:output:09lstm_5/while/lstm_cell_5/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskľ
!lstm_5/while/lstm_cell_5/MatMul_5MatMul"lstm_5/while/lstm_cell_5/mul_1:z:01lstm_5/while/lstm_cell_5/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙´
lstm_5/while/lstm_cell_5/add_1AddV2+lstm_5/while/lstm_cell_5/BiasAdd_1:output:0+lstm_5/while/lstm_cell_5/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"lstm_5/while/lstm_cell_5/Sigmoid_1Sigmoid"lstm_5/while/lstm_cell_5/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/while/lstm_cell_5/mul_4Mul&lstm_5/while/lstm_cell_5/Sigmoid_1:y:0lstm_5_while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)lstm_5/while/lstm_cell_5/ReadVariableOp_2ReadVariableOp2lstm_5_while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
.lstm_5/while/lstm_cell_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
0lstm_5/while/lstm_cell_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
0lstm_5/while/lstm_cell_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(lstm_5/while/lstm_cell_5/strided_slice_2StridedSlice1lstm_5/while/lstm_cell_5/ReadVariableOp_2:value:07lstm_5/while/lstm_cell_5/strided_slice_2/stack:output:09lstm_5/while/lstm_cell_5/strided_slice_2/stack_1:output:09lstm_5/while/lstm_cell_5/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskľ
!lstm_5/while/lstm_cell_5/MatMul_6MatMul"lstm_5/while/lstm_cell_5/mul_2:z:01lstm_5/while/lstm_cell_5/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙´
lstm_5/while/lstm_cell_5/add_2AddV2+lstm_5/while/lstm_cell_5/BiasAdd_2:output:0+lstm_5/while/lstm_cell_5/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_5/while/lstm_cell_5/TanhTanh"lstm_5/while/lstm_cell_5/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
lstm_5/while/lstm_cell_5/mul_5Mul$lstm_5/while/lstm_cell_5/Sigmoid:y:0!lstm_5/while/lstm_cell_5/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_5/while/lstm_cell_5/add_3AddV2"lstm_5/while/lstm_cell_5/mul_4:z:0"lstm_5/while/lstm_cell_5/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)lstm_5/while/lstm_cell_5/ReadVariableOp_3ReadVariableOp2lstm_5_while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
.lstm_5/while/lstm_cell_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
0lstm_5/while/lstm_cell_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0lstm_5/while/lstm_cell_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(lstm_5/while/lstm_cell_5/strided_slice_3StridedSlice1lstm_5/while/lstm_cell_5/ReadVariableOp_3:value:07lstm_5/while/lstm_cell_5/strided_slice_3/stack:output:09lstm_5/while/lstm_cell_5/strided_slice_3/stack_1:output:09lstm_5/while/lstm_cell_5/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskľ
!lstm_5/while/lstm_cell_5/MatMul_7MatMul"lstm_5/while/lstm_cell_5/mul_3:z:01lstm_5/while/lstm_cell_5/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙´
lstm_5/while/lstm_cell_5/add_4AddV2+lstm_5/while/lstm_cell_5/BiasAdd_3:output:0+lstm_5/while/lstm_cell_5/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"lstm_5/while/lstm_cell_5/Sigmoid_2Sigmoid"lstm_5/while/lstm_cell_5/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_5/while/lstm_cell_5/Tanh_1Tanh"lstm_5/while/lstm_cell_5/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ľ
lstm_5/while/lstm_cell_5/mul_6Mul&lstm_5/while/lstm_cell_5/Sigmoid_2:y:0#lstm_5/while/lstm_cell_5/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ŕ
1lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_5_while_placeholder_1lstm_5_while_placeholder"lstm_5/while/lstm_cell_5/mul_6:z:0*
_output_shapes
: *
element_dtype0:éčŇT
lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_5/while/addAddV2lstm_5_while_placeholderlstm_5/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_5/while/add_1AddV2&lstm_5_while_lstm_5_while_loop_counterlstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_5/while/IdentityIdentitylstm_5/while/add_1:z:0^lstm_5/while/NoOp*
T0*
_output_shapes
: 
lstm_5/while/Identity_1Identity,lstm_5_while_lstm_5_while_maximum_iterations^lstm_5/while/NoOp*
T0*
_output_shapes
: n
lstm_5/while/Identity_2Identitylstm_5/while/add:z:0^lstm_5/while/NoOp*
T0*
_output_shapes
: Ž
lstm_5/while/Identity_3IdentityAlstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_5/while/NoOp*
T0*
_output_shapes
: :éčŇ
lstm_5/while/Identity_4Identity"lstm_5/while/lstm_cell_5/mul_6:z:0^lstm_5/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_5/while/Identity_5Identity"lstm_5/while/lstm_cell_5/add_3:z:0^lstm_5/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ă
lstm_5/while/NoOpNoOp(^lstm_5/while/lstm_cell_5/ReadVariableOp*^lstm_5/while/lstm_cell_5/ReadVariableOp_1*^lstm_5/while/lstm_cell_5/ReadVariableOp_2*^lstm_5/while/lstm_cell_5/ReadVariableOp_3.^lstm_5/while/lstm_cell_5/split/ReadVariableOp0^lstm_5/while/lstm_cell_5/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_5_while_identitylstm_5/while/Identity:output:0";
lstm_5_while_identity_1 lstm_5/while/Identity_1:output:0";
lstm_5_while_identity_2 lstm_5/while/Identity_2:output:0";
lstm_5_while_identity_3 lstm_5/while/Identity_3:output:0";
lstm_5_while_identity_4 lstm_5/while/Identity_4:output:0";
lstm_5_while_identity_5 lstm_5/while/Identity_5:output:0"L
#lstm_5_while_lstm_5_strided_slice_1%lstm_5_while_lstm_5_strided_slice_1_0"f
0lstm_5_while_lstm_cell_5_readvariableop_resource2lstm_5_while_lstm_cell_5_readvariableop_resource_0"v
8lstm_5_while_lstm_cell_5_split_1_readvariableop_resource:lstm_5_while_lstm_cell_5_split_1_readvariableop_resource_0"r
6lstm_5_while_lstm_cell_5_split_readvariableop_resource8lstm_5_while_lstm_cell_5_split_readvariableop_resource_0"Ä
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensoralstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2R
'lstm_5/while/lstm_cell_5/ReadVariableOp'lstm_5/while/lstm_cell_5/ReadVariableOp2V
)lstm_5/while/lstm_cell_5/ReadVariableOp_1)lstm_5/while/lstm_cell_5/ReadVariableOp_12V
)lstm_5/while/lstm_cell_5/ReadVariableOp_2)lstm_5/while/lstm_cell_5/ReadVariableOp_22V
)lstm_5/while/lstm_cell_5/ReadVariableOp_3)lstm_5/while/lstm_cell_5/ReadVariableOp_32^
-lstm_5/while/lstm_cell_5/split/ReadVariableOp-lstm_5/while/lstm_cell_5/split/ReadVariableOp2b
/lstm_5/while/lstm_cell_5/split_1/ReadVariableOp/lstm_5/while/lstm_cell_5/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 
Ü
 
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_376790

inputs9
&dense_2_matmul_readvariableop_resource:	95
'dense_2_biasadd_readvariableop_resource:9
identity˘dense_2/BiasAdd/ReadVariableOp˘dense_2/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	9*
dtype0
dense_2/MatMulMatMulReshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:9*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9f
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :9
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapedense_2/Softmax:softmax:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
÷
÷
,__inference_lstm_cell_5_layer_call_fn_377062

inputs
states_0
states_1
unknown:

	unknown_0:	
	unknown_1:

identity

identity_1

identity_2˘StatefulPartitionedCallŞ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_5_layer_call_and_return_conditional_losses_371934p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states/1
Ť
ô
H__inference_sequential_2_layer_call_and_return_conditional_losses_372640

inputs 
lstm_4_372389:	9
lstm_4_372391:	!
lstm_4_372393:
!
lstm_5_372625:

lstm_5_372627:	!
lstm_5_372629:
,
time_distributed_2_372632:	9'
time_distributed_2_372634:9
identity˘lstm_4/StatefulPartitionedCall˘lstm_5/StatefulPartitionedCall˘*time_distributed_2/StatefulPartitionedCall
lstm_4/StatefulPartitionedCallStatefulPartitionedCallinputslstm_4_372389lstm_4_372391lstm_4_372393*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_372388¨
lstm_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0lstm_5_372625lstm_5_372627lstm_5_372629*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_372624Ć
*time_distributed_2/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0time_distributed_2_372632time_distributed_2_372634*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_372107q
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ź
time_distributed_2/ReshapeReshape'lstm_5/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
IdentityIdentity3time_distributed_2/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9ľ
NoOpNoOp^lstm_4/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall+^time_distributed_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : : : : : : 2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall2X
*time_distributed_2/StatefulPartitionedCall*time_distributed_2/StatefulPartitionedCall:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
 
_user_specified_nameinputs
F
˘
__inference__traced_save_377380
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_lstm_4_lstm_cell_4_kernel_read_readvariableopB
>savev2_lstm_4_lstm_cell_4_recurrent_kernel_read_readvariableop6
2savev2_lstm_4_lstm_cell_4_bias_read_readvariableop8
4savev2_lstm_5_lstm_cell_5_kernel_read_readvariableopB
>savev2_lstm_5_lstm_cell_5_recurrent_kernel_read_readvariableop6
2savev2_lstm_5_lstm_cell_5_bias_read_readvariableop8
4savev2_time_distributed_2_kernel_read_readvariableop6
2savev2_time_distributed_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop?
;savev2_adam_lstm_4_lstm_cell_4_kernel_m_read_readvariableopI
Esavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_4_lstm_cell_4_bias_m_read_readvariableop?
;savev2_adam_lstm_5_lstm_cell_5_kernel_m_read_readvariableopI
Esavev2_adam_lstm_5_lstm_cell_5_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_5_lstm_cell_5_bias_m_read_readvariableop?
;savev2_adam_time_distributed_2_kernel_m_read_readvariableop=
9savev2_adam_time_distributed_2_bias_m_read_readvariableop?
;savev2_adam_lstm_4_lstm_cell_4_kernel_v_read_readvariableopI
Esavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_4_lstm_cell_4_bias_v_read_readvariableop?
;savev2_adam_lstm_5_lstm_cell_5_kernel_v_read_readvariableopI
Esavev2_adam_lstm_5_lstm_cell_5_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_5_lstm_cell_5_bias_v_read_readvariableop?
;savev2_adam_time_distributed_2_kernel_v_read_readvariableop=
9savev2_adam_time_distributed_2_bias_v_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ű
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueúB÷ B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH­
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_lstm_4_lstm_cell_4_kernel_read_readvariableop>savev2_lstm_4_lstm_cell_4_recurrent_kernel_read_readvariableop2savev2_lstm_4_lstm_cell_4_bias_read_readvariableop4savev2_lstm_5_lstm_cell_5_kernel_read_readvariableop>savev2_lstm_5_lstm_cell_5_recurrent_kernel_read_readvariableop2savev2_lstm_5_lstm_cell_5_bias_read_readvariableop4savev2_time_distributed_2_kernel_read_readvariableop2savev2_time_distributed_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop;savev2_adam_lstm_4_lstm_cell_4_kernel_m_read_readvariableopEsavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_4_lstm_cell_4_bias_m_read_readvariableop;savev2_adam_lstm_5_lstm_cell_5_kernel_m_read_readvariableopEsavev2_adam_lstm_5_lstm_cell_5_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_5_lstm_cell_5_bias_m_read_readvariableop;savev2_adam_time_distributed_2_kernel_m_read_readvariableop9savev2_adam_time_distributed_2_bias_m_read_readvariableop;savev2_adam_lstm_4_lstm_cell_4_kernel_v_read_readvariableopEsavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_4_lstm_cell_4_bias_v_read_readvariableop;savev2_adam_lstm_5_lstm_cell_5_kernel_v_read_readvariableopEsavev2_adam_lstm_5_lstm_cell_5_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_5_lstm_cell_5_bias_v_read_readvariableop;savev2_adam_time_distributed_2_kernel_v_read_readvariableop9savev2_adam_time_distributed_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes˙
ü: : : : : : :	9:
::
:
::	9:9: : :	9:
::
:
::	9:9:	9:
::
:
::	9:9: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	9:&"
 
_output_shapes
:
:!

_output_shapes	
::&	"
 
_output_shapes
:
:&
"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	9: 

_output_shapes
:9:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	9:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	9: 

_output_shapes
:9:%!

_output_shapes
:	9:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	9: 

_output_shapes
:9: 

_output_shapes
: 
š
Ă
while_cond_372260
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_372260___redundant_placeholder04
0while_while_cond_372260___redundant_placeholder14
0while_while_cond_372260___redundant_placeholder24
0while_while_cond_372260___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:
8

B__inference_lstm_4_layer_call_and_return_conditional_losses_371594

inputs%
lstm_cell_4_371512:	9!
lstm_cell_4_371514:	&
lstm_cell_4_371516:

identity˘#lstm_cell_4/StatefulPartitionedCall˘while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
shrink_axis_maskó
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4_371512lstm_cell_4_371514lstm_cell_4_371516*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_371466n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¸
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4_371512lstm_cell_4_371514lstm_cell_4_371516*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_371525*
condR
while_cond_371524*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙t
NoOpNoOp$^lstm_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : 2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
 
_user_specified_nameinputs
š
Ă
while_cond_371726
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_371726___redundant_placeholder04
0while_while_cond_371726___redundant_placeholder14
0while_while_cond_371726___redundant_placeholder24
0while_while_cond_371726___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:
ď"
ŕ
while_body_371993
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_5_372017_0:
)
while_lstm_cell_5_372019_0:	.
while_lstm_cell_5_372021_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_5_372017:
'
while_lstm_cell_5_372019:	,
while_lstm_cell_5_372021:
˘)while/lstm_cell_5/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0ą
)while/lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_5_372017_0while_lstm_cell_5_372019_0while_lstm_cell_5_372021_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_5_layer_call_and_return_conditional_losses_371934Ű
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_5/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇ
while/Identity_4Identity2while/lstm_cell_5/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/Identity_5Identity2while/lstm_cell_5/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x

while/NoOpNoOp*^while/lstm_cell_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_5_372017while_lstm_cell_5_372017_0"6
while_lstm_cell_5_372019while_lstm_cell_5_372019_0"6
while_lstm_cell_5_372021while_lstm_cell_5_372021_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2V
)while/lstm_cell_5/StatefulPartitionedCall)while/lstm_cell_5/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 
z
ß
B__inference_lstm_5_layer_call_and_return_conditional_losses_376457

inputs=
)lstm_cell_5_split_readvariableop_resource:
:
+lstm_cell_5_split_1_readvariableop_resource:	7
#lstm_cell_5_readvariableop_resource:

identity˘lstm_cell_5/ReadVariableOp˘lstm_cell_5/ReadVariableOp_1˘lstm_cell_5/ReadVariableOp_2˘lstm_cell_5/ReadVariableOp_3˘ lstm_cell_5/split/ReadVariableOp˘"lstm_cell_5/split_1/ReadVariableOp˘while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ę
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskY
lstm_cell_5/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell_5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_5/ones_likeFill$lstm_cell_5/ones_like/Shape:output:0$lstm_cell_5/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_5/split/ReadVariableOpReadVariableOp)lstm_cell_5_split_readvariableop_resource* 
_output_shapes
:
*
dtype0Ę
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0(lstm_cell_5/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0lstm_cell_5/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_5/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_5/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_5/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
lstm_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_5/split_1/ReadVariableOpReadVariableOp+lstm_cell_5_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ź
lstm_cell_5/split_1Split&lstm_cell_5/split_1/split_dim:output:0*lstm_cell_5/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell_5/BiasAddBiasAddlstm_cell_5/MatMul:product:0lstm_cell_5/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/BiasAdd_1BiasAddlstm_cell_5/MatMul_1:product:0lstm_cell_5/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/BiasAdd_2BiasAddlstm_cell_5/MatMul_2:product:0lstm_cell_5/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/BiasAdd_3BiasAddlstm_cell_5/MatMul_3:product:0lstm_cell_5/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
lstm_cell_5/mulMulzeros:output:0lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_5/mul_1Mulzeros:output:0lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_5/mul_2Mulzeros:output:0lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_5/mul_3Mulzeros:output:0lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOpReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell_5/strided_sliceStridedSlice"lstm_cell_5/ReadVariableOp:value:0(lstm_cell_5/strided_slice/stack:output:0*lstm_cell_5/strided_slice/stack_1:output:0*lstm_cell_5/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_4MatMullstm_cell_5/mul:z:0"lstm_cell_5/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/addAddV2lstm_cell_5/BiasAdd:output:0lstm_cell_5/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell_5/SigmoidSigmoidlstm_cell_5/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOp_1ReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_5/strided_slice_1StridedSlice$lstm_cell_5/ReadVariableOp_1:value:0*lstm_cell_5/strided_slice_1/stack:output:0,lstm_cell_5/strided_slice_1/stack_1:output:0,lstm_cell_5/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_5MatMullstm_cell_5/mul_1:z:0$lstm_cell_5/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/add_1AddV2lstm_cell_5/BiasAdd_1:output:0lstm_cell_5/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_cell_5/mul_4Mullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOp_2ReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_5/strided_slice_2StridedSlice$lstm_cell_5/ReadVariableOp_2:value:0*lstm_cell_5/strided_slice_2/stack:output:0,lstm_cell_5/strided_slice_2/stack_1:output:0,lstm_cell_5/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_6MatMullstm_cell_5/mul_2:z:0$lstm_cell_5/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/add_2AddV2lstm_cell_5/BiasAdd_2:output:0lstm_cell_5/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
lstm_cell_5/TanhTanhlstm_cell_5/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
lstm_cell_5/mul_5Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_5/add_3AddV2lstm_cell_5/mul_4:z:0lstm_cell_5/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOp_3ReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_5/strided_slice_3StridedSlice$lstm_cell_5/ReadVariableOp_3:value:0*lstm_cell_5/strided_slice_3/stack:output:0,lstm_cell_5/strided_slice_3/stack_1:output:0,lstm_cell_5/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_7MatMullstm_cell_5/mul_3:z:0$lstm_cell_5/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/add_4AddV2lstm_cell_5/BiasAdd_3:output:0lstm_cell_5/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
lstm_cell_5/Tanh_1Tanhlstm_cell_5/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell_5/mul_6Mullstm_cell_5/Sigmoid_2:y:0lstm_cell_5/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ů
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_5_split_readvariableop_resource+lstm_cell_5_split_1_readvariableop_resource#lstm_cell_5_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_376330*
condR
while_cond_376329*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp^lstm_cell_5/ReadVariableOp^lstm_cell_5/ReadVariableOp_1^lstm_cell_5/ReadVariableOp_2^lstm_cell_5/ReadVariableOp_3!^lstm_cell_5/split/ReadVariableOp#^lstm_cell_5/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 28
lstm_cell_5/ReadVariableOplstm_cell_5/ReadVariableOp2<
lstm_cell_5/ReadVariableOp_1lstm_cell_5/ReadVariableOp_12<
lstm_cell_5/ReadVariableOp_2lstm_cell_5/ReadVariableOp_22<
lstm_cell_5/ReadVariableOp_3lstm_cell_5/ReadVariableOp_32D
 lstm_cell_5/split/ReadVariableOp lstm_cell_5/split/ReadVariableOp2H
"lstm_cell_5/split_1/ReadVariableOp"lstm_cell_5/split_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ă

!__inference__wrapped_model_371135
lstm_4_inputP
=sequential_2_lstm_4_lstm_cell_4_split_readvariableop_resource:	9N
?sequential_2_lstm_4_lstm_cell_4_split_1_readvariableop_resource:	K
7sequential_2_lstm_4_lstm_cell_4_readvariableop_resource:
Q
=sequential_2_lstm_5_lstm_cell_5_split_readvariableop_resource:
N
?sequential_2_lstm_5_lstm_cell_5_split_1_readvariableop_resource:	K
7sequential_2_lstm_5_lstm_cell_5_readvariableop_resource:
Y
Fsequential_2_time_distributed_2_dense_2_matmul_readvariableop_resource:	9U
Gsequential_2_time_distributed_2_dense_2_biasadd_readvariableop_resource:9
identity˘.sequential_2/lstm_4/lstm_cell_4/ReadVariableOp˘0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_1˘0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_2˘0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_3˘4sequential_2/lstm_4/lstm_cell_4/split/ReadVariableOp˘6sequential_2/lstm_4/lstm_cell_4/split_1/ReadVariableOp˘sequential_2/lstm_4/while˘.sequential_2/lstm_5/lstm_cell_5/ReadVariableOp˘0sequential_2/lstm_5/lstm_cell_5/ReadVariableOp_1˘0sequential_2/lstm_5/lstm_cell_5/ReadVariableOp_2˘0sequential_2/lstm_5/lstm_cell_5/ReadVariableOp_3˘4sequential_2/lstm_5/lstm_cell_5/split/ReadVariableOp˘6sequential_2/lstm_5/lstm_cell_5/split_1/ReadVariableOp˘sequential_2/lstm_5/while˘>sequential_2/time_distributed_2/dense_2/BiasAdd/ReadVariableOp˘=sequential_2/time_distributed_2/dense_2/MatMul/ReadVariableOpU
sequential_2/lstm_4/ShapeShapelstm_4_input*
T0*
_output_shapes
:q
'sequential_2/lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_2/lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_2/lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!sequential_2/lstm_4/strided_sliceStridedSlice"sequential_2/lstm_4/Shape:output:00sequential_2/lstm_4/strided_slice/stack:output:02sequential_2/lstm_4/strided_slice/stack_1:output:02sequential_2/lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"sequential_2/lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ż
 sequential_2/lstm_4/zeros/packedPack*sequential_2/lstm_4/strided_slice:output:0+sequential_2/lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_2/lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Š
sequential_2/lstm_4/zerosFill)sequential_2/lstm_4/zeros/packed:output:0(sequential_2/lstm_4/zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
$sequential_2/lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ł
"sequential_2/lstm_4/zeros_1/packedPack*sequential_2/lstm_4/strided_slice:output:0-sequential_2/lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_2/lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ż
sequential_2/lstm_4/zeros_1Fill+sequential_2/lstm_4/zeros_1/packed:output:0*sequential_2/lstm_4/zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
"sequential_2/lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¤
sequential_2/lstm_4/transpose	Transposelstm_4_input+sequential_2/lstm_4/transpose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9l
sequential_2/lstm_4/Shape_1Shape!sequential_2/lstm_4/transpose:y:0*
T0*
_output_shapes
:s
)sequential_2/lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_2/lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_2/lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#sequential_2/lstm_4/strided_slice_1StridedSlice$sequential_2/lstm_4/Shape_1:output:02sequential_2/lstm_4/strided_slice_1/stack:output:04sequential_2/lstm_4/strided_slice_1/stack_1:output:04sequential_2/lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_2/lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙đ
!sequential_2/lstm_4/TensorArrayV2TensorListReserve8sequential_2/lstm_4/TensorArrayV2/element_shape:output:0,sequential_2/lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
Isequential_2/lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   
;sequential_2/lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_2/lstm_4/transpose:y:0Rsequential_2/lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇs
)sequential_2/lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_2/lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_2/lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Í
#sequential_2/lstm_4/strided_slice_2StridedSlice!sequential_2/lstm_4/transpose:y:02sequential_2/lstm_4/strided_slice_2/stack:output:04sequential_2/lstm_4/strided_slice_2/stack_1:output:04sequential_2/lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
shrink_axis_mask
/sequential_2/lstm_4/lstm_cell_4/ones_like/ShapeShape"sequential_2/lstm_4/zeros:output:0*
T0*
_output_shapes
:t
/sequential_2/lstm_4/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ř
)sequential_2/lstm_4/lstm_cell_4/ones_likeFill8sequential_2/lstm_4/lstm_cell_4/ones_like/Shape:output:08sequential_2/lstm_4/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
/sequential_2/lstm_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ł
4sequential_2/lstm_4/lstm_cell_4/split/ReadVariableOpReadVariableOp=sequential_2_lstm_4_lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	9*
dtype0
%sequential_2/lstm_4/lstm_cell_4/splitSplit8sequential_2/lstm_4/lstm_cell_4/split/split_dim:output:0<sequential_2/lstm_4/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_splitÁ
&sequential_2/lstm_4/lstm_cell_4/MatMulMatMul,sequential_2/lstm_4/strided_slice_2:output:0.sequential_2/lstm_4/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
(sequential_2/lstm_4/lstm_cell_4/MatMul_1MatMul,sequential_2/lstm_4/strided_slice_2:output:0.sequential_2/lstm_4/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
(sequential_2/lstm_4/lstm_cell_4/MatMul_2MatMul,sequential_2/lstm_4/strided_slice_2:output:0.sequential_2/lstm_4/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
(sequential_2/lstm_4/lstm_cell_4/MatMul_3MatMul,sequential_2/lstm_4/strided_slice_2:output:0.sequential_2/lstm_4/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
1sequential_2/lstm_4/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ł
6sequential_2/lstm_4/lstm_cell_4/split_1/ReadVariableOpReadVariableOp?sequential_2_lstm_4_lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ř
'sequential_2/lstm_4/lstm_cell_4/split_1Split:sequential_2/lstm_4/lstm_cell_4/split_1/split_dim:output:0>sequential_2/lstm_4/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_splitÉ
'sequential_2/lstm_4/lstm_cell_4/BiasAddBiasAdd0sequential_2/lstm_4/lstm_cell_4/MatMul:product:00sequential_2/lstm_4/lstm_cell_4/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Í
)sequential_2/lstm_4/lstm_cell_4/BiasAdd_1BiasAdd2sequential_2/lstm_4/lstm_cell_4/MatMul_1:product:00sequential_2/lstm_4/lstm_cell_4/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Í
)sequential_2/lstm_4/lstm_cell_4/BiasAdd_2BiasAdd2sequential_2/lstm_4/lstm_cell_4/MatMul_2:product:00sequential_2/lstm_4/lstm_cell_4/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Í
)sequential_2/lstm_4/lstm_cell_4/BiasAdd_3BiasAdd2sequential_2/lstm_4/lstm_cell_4/MatMul_3:product:00sequential_2/lstm_4/lstm_cell_4/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ
#sequential_2/lstm_4/lstm_cell_4/mulMul"sequential_2/lstm_4/zeros:output:02sequential_2/lstm_4/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ˇ
%sequential_2/lstm_4/lstm_cell_4/mul_1Mul"sequential_2/lstm_4/zeros:output:02sequential_2/lstm_4/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ˇ
%sequential_2/lstm_4/lstm_cell_4/mul_2Mul"sequential_2/lstm_4/zeros:output:02sequential_2/lstm_4/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ˇ
%sequential_2/lstm_4/lstm_cell_4/mul_3Mul"sequential_2/lstm_4/zeros:output:02sequential_2/lstm_4/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¨
.sequential_2/lstm_4/lstm_cell_4/ReadVariableOpReadVariableOp7sequential_2_lstm_4_lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0
3sequential_2/lstm_4/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
5sequential_2/lstm_4/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
5sequential_2/lstm_4/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
-sequential_2/lstm_4/lstm_cell_4/strided_sliceStridedSlice6sequential_2/lstm_4/lstm_cell_4/ReadVariableOp:value:0<sequential_2/lstm_4/lstm_cell_4/strided_slice/stack:output:0>sequential_2/lstm_4/lstm_cell_4/strided_slice/stack_1:output:0>sequential_2/lstm_4/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskĆ
(sequential_2/lstm_4/lstm_cell_4/MatMul_4MatMul'sequential_2/lstm_4/lstm_cell_4/mul:z:06sequential_2/lstm_4/lstm_cell_4/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
#sequential_2/lstm_4/lstm_cell_4/addAddV20sequential_2/lstm_4/lstm_cell_4/BiasAdd:output:02sequential_2/lstm_4/lstm_cell_4/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
'sequential_2/lstm_4/lstm_cell_4/SigmoidSigmoid'sequential_2/lstm_4/lstm_cell_4/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ş
0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_1ReadVariableOp7sequential_2_lstm_4_lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0
5sequential_2/lstm_4/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
7sequential_2/lstm_4/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
7sequential_2/lstm_4/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
/sequential_2/lstm_4/lstm_cell_4/strided_slice_1StridedSlice8sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_1:value:0>sequential_2/lstm_4/lstm_cell_4/strided_slice_1/stack:output:0@sequential_2/lstm_4/lstm_cell_4/strided_slice_1/stack_1:output:0@sequential_2/lstm_4/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskĘ
(sequential_2/lstm_4/lstm_cell_4/MatMul_5MatMul)sequential_2/lstm_4/lstm_cell_4/mul_1:z:08sequential_2/lstm_4/lstm_cell_4/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙É
%sequential_2/lstm_4/lstm_cell_4/add_1AddV22sequential_2/lstm_4/lstm_cell_4/BiasAdd_1:output:02sequential_2/lstm_4/lstm_cell_4/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)sequential_2/lstm_4/lstm_cell_4/Sigmoid_1Sigmoid)sequential_2/lstm_4/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙´
%sequential_2/lstm_4/lstm_cell_4/mul_4Mul-sequential_2/lstm_4/lstm_cell_4/Sigmoid_1:y:0$sequential_2/lstm_4/zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ş
0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_2ReadVariableOp7sequential_2_lstm_4_lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0
5sequential_2/lstm_4/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
7sequential_2/lstm_4/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
7sequential_2/lstm_4/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
/sequential_2/lstm_4/lstm_cell_4/strided_slice_2StridedSlice8sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_2:value:0>sequential_2/lstm_4/lstm_cell_4/strided_slice_2/stack:output:0@sequential_2/lstm_4/lstm_cell_4/strided_slice_2/stack_1:output:0@sequential_2/lstm_4/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskĘ
(sequential_2/lstm_4/lstm_cell_4/MatMul_6MatMul)sequential_2/lstm_4/lstm_cell_4/mul_2:z:08sequential_2/lstm_4/lstm_cell_4/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙É
%sequential_2/lstm_4/lstm_cell_4/add_2AddV22sequential_2/lstm_4/lstm_cell_4/BiasAdd_2:output:02sequential_2/lstm_4/lstm_cell_4/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
$sequential_2/lstm_4/lstm_cell_4/TanhTanh)sequential_2/lstm_4/lstm_cell_4/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ś
%sequential_2/lstm_4/lstm_cell_4/mul_5Mul+sequential_2/lstm_4/lstm_cell_4/Sigmoid:y:0(sequential_2/lstm_4/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ˇ
%sequential_2/lstm_4/lstm_cell_4/add_3AddV2)sequential_2/lstm_4/lstm_cell_4/mul_4:z:0)sequential_2/lstm_4/lstm_cell_4/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ş
0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_3ReadVariableOp7sequential_2_lstm_4_lstm_cell_4_readvariableop_resource* 
_output_shapes
:
*
dtype0
5sequential_2/lstm_4/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
7sequential_2/lstm_4/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
7sequential_2/lstm_4/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
/sequential_2/lstm_4/lstm_cell_4/strided_slice_3StridedSlice8sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_3:value:0>sequential_2/lstm_4/lstm_cell_4/strided_slice_3/stack:output:0@sequential_2/lstm_4/lstm_cell_4/strided_slice_3/stack_1:output:0@sequential_2/lstm_4/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskĘ
(sequential_2/lstm_4/lstm_cell_4/MatMul_7MatMul)sequential_2/lstm_4/lstm_cell_4/mul_3:z:08sequential_2/lstm_4/lstm_cell_4/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙É
%sequential_2/lstm_4/lstm_cell_4/add_4AddV22sequential_2/lstm_4/lstm_cell_4/BiasAdd_3:output:02sequential_2/lstm_4/lstm_cell_4/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)sequential_2/lstm_4/lstm_cell_4/Sigmoid_2Sigmoid)sequential_2/lstm_4/lstm_cell_4/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
&sequential_2/lstm_4/lstm_cell_4/Tanh_1Tanh)sequential_2/lstm_4/lstm_cell_4/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ş
%sequential_2/lstm_4/lstm_cell_4/mul_6Mul-sequential_2/lstm_4/lstm_cell_4/Sigmoid_2:y:0*sequential_2/lstm_4/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
1sequential_2/lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ô
#sequential_2/lstm_4/TensorArrayV2_1TensorListReserve:sequential_2/lstm_4/TensorArrayV2_1/element_shape:output:0,sequential_2/lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇZ
sequential_2/lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_2/lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙h
&sequential_2/lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
sequential_2/lstm_4/whileWhile/sequential_2/lstm_4/while/loop_counter:output:05sequential_2/lstm_4/while/maximum_iterations:output:0!sequential_2/lstm_4/time:output:0,sequential_2/lstm_4/TensorArrayV2_1:handle:0"sequential_2/lstm_4/zeros:output:0$sequential_2/lstm_4/zeros_1:output:0,sequential_2/lstm_4/strided_slice_1:output:0Ksequential_2/lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0=sequential_2_lstm_4_lstm_cell_4_split_readvariableop_resource?sequential_2_lstm_4_lstm_cell_4_split_1_readvariableop_resource7sequential_2_lstm_4_lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *1
body)R'
%sequential_2_lstm_4_while_body_370763*1
cond)R'
%sequential_2_lstm_4_while_cond_370762*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
Dsequential_2/lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
6sequential_2/lstm_4/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_2/lstm_4/while:output:3Msequential_2/lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0|
)sequential_2/lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙u
+sequential_2/lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_2/lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ě
#sequential_2/lstm_4/strided_slice_3StridedSlice?sequential_2/lstm_4/TensorArrayV2Stack/TensorListStack:tensor:02sequential_2/lstm_4/strided_slice_3/stack:output:04sequential_2/lstm_4/strided_slice_3/stack_1:output:04sequential_2/lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_masky
$sequential_2/lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ü
sequential_2/lstm_4/transpose_1	Transpose?sequential_2/lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_2/lstm_4/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙o
sequential_2/lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
sequential_2/lstm_5/ShapeShape#sequential_2/lstm_4/transpose_1:y:0*
T0*
_output_shapes
:q
'sequential_2/lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_2/lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_2/lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!sequential_2/lstm_5/strided_sliceStridedSlice"sequential_2/lstm_5/Shape:output:00sequential_2/lstm_5/strided_slice/stack:output:02sequential_2/lstm_5/strided_slice/stack_1:output:02sequential_2/lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"sequential_2/lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ż
 sequential_2/lstm_5/zeros/packedPack*sequential_2/lstm_5/strided_slice:output:0+sequential_2/lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_2/lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Š
sequential_2/lstm_5/zerosFill)sequential_2/lstm_5/zeros/packed:output:0(sequential_2/lstm_5/zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
$sequential_2/lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ł
"sequential_2/lstm_5/zeros_1/packedPack*sequential_2/lstm_5/strided_slice:output:0-sequential_2/lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_2/lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ż
sequential_2/lstm_5/zeros_1Fill+sequential_2/lstm_5/zeros_1/packed:output:0*sequential_2/lstm_5/zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
"sequential_2/lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ź
sequential_2/lstm_5/transpose	Transpose#sequential_2/lstm_4/transpose_1:y:0+sequential_2/lstm_5/transpose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙l
sequential_2/lstm_5/Shape_1Shape!sequential_2/lstm_5/transpose:y:0*
T0*
_output_shapes
:s
)sequential_2/lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_2/lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_2/lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#sequential_2/lstm_5/strided_slice_1StridedSlice$sequential_2/lstm_5/Shape_1:output:02sequential_2/lstm_5/strided_slice_1/stack:output:04sequential_2/lstm_5/strided_slice_1/stack_1:output:04sequential_2/lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_2/lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙đ
!sequential_2/lstm_5/TensorArrayV2TensorListReserve8sequential_2/lstm_5/TensorArrayV2/element_shape:output:0,sequential_2/lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
Isequential_2/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
;sequential_2/lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_2/lstm_5/transpose:y:0Rsequential_2/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇs
)sequential_2/lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_2/lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_2/lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
#sequential_2/lstm_5/strided_slice_2StridedSlice!sequential_2/lstm_5/transpose:y:02sequential_2/lstm_5/strided_slice_2/stack:output:04sequential_2/lstm_5/strided_slice_2/stack_1:output:04sequential_2/lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_mask
/sequential_2/lstm_5/lstm_cell_5/ones_like/ShapeShape"sequential_2/lstm_5/zeros:output:0*
T0*
_output_shapes
:t
/sequential_2/lstm_5/lstm_cell_5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ř
)sequential_2/lstm_5/lstm_cell_5/ones_likeFill8sequential_2/lstm_5/lstm_cell_5/ones_like/Shape:output:08sequential_2/lstm_5/lstm_cell_5/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
/sequential_2/lstm_5/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :´
4sequential_2/lstm_5/lstm_cell_5/split/ReadVariableOpReadVariableOp=sequential_2_lstm_5_lstm_cell_5_split_readvariableop_resource* 
_output_shapes
:
*
dtype0
%sequential_2/lstm_5/lstm_cell_5/splitSplit8sequential_2/lstm_5/lstm_cell_5/split/split_dim:output:0<sequential_2/lstm_5/lstm_cell_5/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_splitÁ
&sequential_2/lstm_5/lstm_cell_5/MatMulMatMul,sequential_2/lstm_5/strided_slice_2:output:0.sequential_2/lstm_5/lstm_cell_5/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
(sequential_2/lstm_5/lstm_cell_5/MatMul_1MatMul,sequential_2/lstm_5/strided_slice_2:output:0.sequential_2/lstm_5/lstm_cell_5/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
(sequential_2/lstm_5/lstm_cell_5/MatMul_2MatMul,sequential_2/lstm_5/strided_slice_2:output:0.sequential_2/lstm_5/lstm_cell_5/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
(sequential_2/lstm_5/lstm_cell_5/MatMul_3MatMul,sequential_2/lstm_5/strided_slice_2:output:0.sequential_2/lstm_5/lstm_cell_5/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
1sequential_2/lstm_5/lstm_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ł
6sequential_2/lstm_5/lstm_cell_5/split_1/ReadVariableOpReadVariableOp?sequential_2_lstm_5_lstm_cell_5_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ř
'sequential_2/lstm_5/lstm_cell_5/split_1Split:sequential_2/lstm_5/lstm_cell_5/split_1/split_dim:output:0>sequential_2/lstm_5/lstm_cell_5/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_splitÉ
'sequential_2/lstm_5/lstm_cell_5/BiasAddBiasAdd0sequential_2/lstm_5/lstm_cell_5/MatMul:product:00sequential_2/lstm_5/lstm_cell_5/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Í
)sequential_2/lstm_5/lstm_cell_5/BiasAdd_1BiasAdd2sequential_2/lstm_5/lstm_cell_5/MatMul_1:product:00sequential_2/lstm_5/lstm_cell_5/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Í
)sequential_2/lstm_5/lstm_cell_5/BiasAdd_2BiasAdd2sequential_2/lstm_5/lstm_cell_5/MatMul_2:product:00sequential_2/lstm_5/lstm_cell_5/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Í
)sequential_2/lstm_5/lstm_cell_5/BiasAdd_3BiasAdd2sequential_2/lstm_5/lstm_cell_5/MatMul_3:product:00sequential_2/lstm_5/lstm_cell_5/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ
#sequential_2/lstm_5/lstm_cell_5/mulMul"sequential_2/lstm_5/zeros:output:02sequential_2/lstm_5/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ˇ
%sequential_2/lstm_5/lstm_cell_5/mul_1Mul"sequential_2/lstm_5/zeros:output:02sequential_2/lstm_5/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ˇ
%sequential_2/lstm_5/lstm_cell_5/mul_2Mul"sequential_2/lstm_5/zeros:output:02sequential_2/lstm_5/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ˇ
%sequential_2/lstm_5/lstm_cell_5/mul_3Mul"sequential_2/lstm_5/zeros:output:02sequential_2/lstm_5/lstm_cell_5/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¨
.sequential_2/lstm_5/lstm_cell_5/ReadVariableOpReadVariableOp7sequential_2_lstm_5_lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0
3sequential_2/lstm_5/lstm_cell_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
5sequential_2/lstm_5/lstm_cell_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
5sequential_2/lstm_5/lstm_cell_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
-sequential_2/lstm_5/lstm_cell_5/strided_sliceStridedSlice6sequential_2/lstm_5/lstm_cell_5/ReadVariableOp:value:0<sequential_2/lstm_5/lstm_cell_5/strided_slice/stack:output:0>sequential_2/lstm_5/lstm_cell_5/strided_slice/stack_1:output:0>sequential_2/lstm_5/lstm_cell_5/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskĆ
(sequential_2/lstm_5/lstm_cell_5/MatMul_4MatMul'sequential_2/lstm_5/lstm_cell_5/mul:z:06sequential_2/lstm_5/lstm_cell_5/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
#sequential_2/lstm_5/lstm_cell_5/addAddV20sequential_2/lstm_5/lstm_cell_5/BiasAdd:output:02sequential_2/lstm_5/lstm_cell_5/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
'sequential_2/lstm_5/lstm_cell_5/SigmoidSigmoid'sequential_2/lstm_5/lstm_cell_5/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ş
0sequential_2/lstm_5/lstm_cell_5/ReadVariableOp_1ReadVariableOp7sequential_2_lstm_5_lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0
5sequential_2/lstm_5/lstm_cell_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
7sequential_2/lstm_5/lstm_cell_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
7sequential_2/lstm_5/lstm_cell_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
/sequential_2/lstm_5/lstm_cell_5/strided_slice_1StridedSlice8sequential_2/lstm_5/lstm_cell_5/ReadVariableOp_1:value:0>sequential_2/lstm_5/lstm_cell_5/strided_slice_1/stack:output:0@sequential_2/lstm_5/lstm_cell_5/strided_slice_1/stack_1:output:0@sequential_2/lstm_5/lstm_cell_5/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskĘ
(sequential_2/lstm_5/lstm_cell_5/MatMul_5MatMul)sequential_2/lstm_5/lstm_cell_5/mul_1:z:08sequential_2/lstm_5/lstm_cell_5/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙É
%sequential_2/lstm_5/lstm_cell_5/add_1AddV22sequential_2/lstm_5/lstm_cell_5/BiasAdd_1:output:02sequential_2/lstm_5/lstm_cell_5/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)sequential_2/lstm_5/lstm_cell_5/Sigmoid_1Sigmoid)sequential_2/lstm_5/lstm_cell_5/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙´
%sequential_2/lstm_5/lstm_cell_5/mul_4Mul-sequential_2/lstm_5/lstm_cell_5/Sigmoid_1:y:0$sequential_2/lstm_5/zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ş
0sequential_2/lstm_5/lstm_cell_5/ReadVariableOp_2ReadVariableOp7sequential_2_lstm_5_lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0
5sequential_2/lstm_5/lstm_cell_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
7sequential_2/lstm_5/lstm_cell_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
7sequential_2/lstm_5/lstm_cell_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
/sequential_2/lstm_5/lstm_cell_5/strided_slice_2StridedSlice8sequential_2/lstm_5/lstm_cell_5/ReadVariableOp_2:value:0>sequential_2/lstm_5/lstm_cell_5/strided_slice_2/stack:output:0@sequential_2/lstm_5/lstm_cell_5/strided_slice_2/stack_1:output:0@sequential_2/lstm_5/lstm_cell_5/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskĘ
(sequential_2/lstm_5/lstm_cell_5/MatMul_6MatMul)sequential_2/lstm_5/lstm_cell_5/mul_2:z:08sequential_2/lstm_5/lstm_cell_5/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙É
%sequential_2/lstm_5/lstm_cell_5/add_2AddV22sequential_2/lstm_5/lstm_cell_5/BiasAdd_2:output:02sequential_2/lstm_5/lstm_cell_5/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
$sequential_2/lstm_5/lstm_cell_5/TanhTanh)sequential_2/lstm_5/lstm_cell_5/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ś
%sequential_2/lstm_5/lstm_cell_5/mul_5Mul+sequential_2/lstm_5/lstm_cell_5/Sigmoid:y:0(sequential_2/lstm_5/lstm_cell_5/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ˇ
%sequential_2/lstm_5/lstm_cell_5/add_3AddV2)sequential_2/lstm_5/lstm_cell_5/mul_4:z:0)sequential_2/lstm_5/lstm_cell_5/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ş
0sequential_2/lstm_5/lstm_cell_5/ReadVariableOp_3ReadVariableOp7sequential_2_lstm_5_lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0
5sequential_2/lstm_5/lstm_cell_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
7sequential_2/lstm_5/lstm_cell_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
7sequential_2/lstm_5/lstm_cell_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
/sequential_2/lstm_5/lstm_cell_5/strided_slice_3StridedSlice8sequential_2/lstm_5/lstm_cell_5/ReadVariableOp_3:value:0>sequential_2/lstm_5/lstm_cell_5/strided_slice_3/stack:output:0@sequential_2/lstm_5/lstm_cell_5/strided_slice_3/stack_1:output:0@sequential_2/lstm_5/lstm_cell_5/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskĘ
(sequential_2/lstm_5/lstm_cell_5/MatMul_7MatMul)sequential_2/lstm_5/lstm_cell_5/mul_3:z:08sequential_2/lstm_5/lstm_cell_5/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙É
%sequential_2/lstm_5/lstm_cell_5/add_4AddV22sequential_2/lstm_5/lstm_cell_5/BiasAdd_3:output:02sequential_2/lstm_5/lstm_cell_5/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)sequential_2/lstm_5/lstm_cell_5/Sigmoid_2Sigmoid)sequential_2/lstm_5/lstm_cell_5/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
&sequential_2/lstm_5/lstm_cell_5/Tanh_1Tanh)sequential_2/lstm_5/lstm_cell_5/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ş
%sequential_2/lstm_5/lstm_cell_5/mul_6Mul-sequential_2/lstm_5/lstm_cell_5/Sigmoid_2:y:0*sequential_2/lstm_5/lstm_cell_5/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
1sequential_2/lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ô
#sequential_2/lstm_5/TensorArrayV2_1TensorListReserve:sequential_2/lstm_5/TensorArrayV2_1/element_shape:output:0,sequential_2/lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇZ
sequential_2/lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_2/lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙h
&sequential_2/lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
sequential_2/lstm_5/whileWhile/sequential_2/lstm_5/while/loop_counter:output:05sequential_2/lstm_5/while/maximum_iterations:output:0!sequential_2/lstm_5/time:output:0,sequential_2/lstm_5/TensorArrayV2_1:handle:0"sequential_2/lstm_5/zeros:output:0$sequential_2/lstm_5/zeros_1:output:0,sequential_2/lstm_5/strided_slice_1:output:0Ksequential_2/lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0=sequential_2_lstm_5_lstm_cell_5_split_readvariableop_resource?sequential_2_lstm_5_lstm_cell_5_split_1_readvariableop_resource7sequential_2_lstm_5_lstm_cell_5_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *1
body)R'
%sequential_2_lstm_5_while_body_370988*1
cond)R'
%sequential_2_lstm_5_while_cond_370987*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
Dsequential_2/lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
6sequential_2/lstm_5/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_2/lstm_5/while:output:3Msequential_2/lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0|
)sequential_2/lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙u
+sequential_2/lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_2/lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ě
#sequential_2/lstm_5/strided_slice_3StridedSlice?sequential_2/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:02sequential_2/lstm_5/strided_slice_3/stack:output:04sequential_2/lstm_5/strided_slice_3/stack_1:output:04sequential_2/lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_masky
$sequential_2/lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ü
sequential_2/lstm_5/transpose_1	Transpose?sequential_2/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_2/lstm_5/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙o
sequential_2/lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    x
%sequential_2/time_distributed_2/ShapeShape#sequential_2/lstm_5/transpose_1:y:0*
T0*
_output_shapes
:}
3sequential_2/time_distributed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
5sequential_2/time_distributed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_2/time_distributed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ń
-sequential_2/time_distributed_2/strided_sliceStridedSlice.sequential_2/time_distributed_2/Shape:output:0<sequential_2/time_distributed_2/strided_slice/stack:output:0>sequential_2/time_distributed_2/strided_slice/stack_1:output:0>sequential_2/time_distributed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
-sequential_2/time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Â
'sequential_2/time_distributed_2/ReshapeReshape#sequential_2/lstm_5/transpose_1:y:06sequential_2/time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
=sequential_2/time_distributed_2/dense_2/MatMul/ReadVariableOpReadVariableOpFsequential_2_time_distributed_2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	9*
dtype0ă
.sequential_2/time_distributed_2/dense_2/MatMulMatMul0sequential_2/time_distributed_2/Reshape:output:0Esequential_2/time_distributed_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9Â
>sequential_2/time_distributed_2/dense_2/BiasAdd/ReadVariableOpReadVariableOpGsequential_2_time_distributed_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:9*
dtype0î
/sequential_2/time_distributed_2/dense_2/BiasAddBiasAdd8sequential_2/time_distributed_2/dense_2/MatMul:product:0Fsequential_2/time_distributed_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9Ś
/sequential_2/time_distributed_2/dense_2/SoftmaxSoftmax8sequential_2/time_distributed_2/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9|
1sequential_2/time_distributed_2/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙s
1sequential_2/time_distributed_2/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :9
/sequential_2/time_distributed_2/Reshape_1/shapePack:sequential_2/time_distributed_2/Reshape_1/shape/0:output:06sequential_2/time_distributed_2/strided_slice:output:0:sequential_2/time_distributed_2/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:č
)sequential_2/time_distributed_2/Reshape_1Reshape9sequential_2/time_distributed_2/dense_2/Softmax:softmax:08sequential_2/time_distributed_2/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
/sequential_2/time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ć
)sequential_2/time_distributed_2/Reshape_2Reshape#sequential_2/lstm_5/transpose_1:y:08sequential_2/time_distributed_2/Reshape_2/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
IdentityIdentity2sequential_2/time_distributed_2/Reshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9ó
NoOpNoOp/^sequential_2/lstm_4/lstm_cell_4/ReadVariableOp1^sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_11^sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_21^sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_35^sequential_2/lstm_4/lstm_cell_4/split/ReadVariableOp7^sequential_2/lstm_4/lstm_cell_4/split_1/ReadVariableOp^sequential_2/lstm_4/while/^sequential_2/lstm_5/lstm_cell_5/ReadVariableOp1^sequential_2/lstm_5/lstm_cell_5/ReadVariableOp_11^sequential_2/lstm_5/lstm_cell_5/ReadVariableOp_21^sequential_2/lstm_5/lstm_cell_5/ReadVariableOp_35^sequential_2/lstm_5/lstm_cell_5/split/ReadVariableOp7^sequential_2/lstm_5/lstm_cell_5/split_1/ReadVariableOp^sequential_2/lstm_5/while?^sequential_2/time_distributed_2/dense_2/BiasAdd/ReadVariableOp>^sequential_2/time_distributed_2/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : : : : : : 2`
.sequential_2/lstm_4/lstm_cell_4/ReadVariableOp.sequential_2/lstm_4/lstm_cell_4/ReadVariableOp2d
0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_10sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_12d
0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_20sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_22d
0sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_30sequential_2/lstm_4/lstm_cell_4/ReadVariableOp_32l
4sequential_2/lstm_4/lstm_cell_4/split/ReadVariableOp4sequential_2/lstm_4/lstm_cell_4/split/ReadVariableOp2p
6sequential_2/lstm_4/lstm_cell_4/split_1/ReadVariableOp6sequential_2/lstm_4/lstm_cell_4/split_1/ReadVariableOp26
sequential_2/lstm_4/whilesequential_2/lstm_4/while2`
.sequential_2/lstm_5/lstm_cell_5/ReadVariableOp.sequential_2/lstm_5/lstm_cell_5/ReadVariableOp2d
0sequential_2/lstm_5/lstm_cell_5/ReadVariableOp_10sequential_2/lstm_5/lstm_cell_5/ReadVariableOp_12d
0sequential_2/lstm_5/lstm_cell_5/ReadVariableOp_20sequential_2/lstm_5/lstm_cell_5/ReadVariableOp_22d
0sequential_2/lstm_5/lstm_cell_5/ReadVariableOp_30sequential_2/lstm_5/lstm_cell_5/ReadVariableOp_32l
4sequential_2/lstm_5/lstm_cell_5/split/ReadVariableOp4sequential_2/lstm_5/lstm_cell_5/split/ReadVariableOp2p
6sequential_2/lstm_5/lstm_cell_5/split_1/ReadVariableOp6sequential_2/lstm_5/lstm_cell_5/split_1/ReadVariableOp26
sequential_2/lstm_5/whilesequential_2/lstm_5/while2
>sequential_2/time_distributed_2/dense_2/BiasAdd/ReadVariableOp>sequential_2/time_distributed_2/dense_2/BiasAdd/ReadVariableOp2~
=sequential_2/time_distributed_2/dense_2/MatMul/ReadVariableOp=sequential_2/time_distributed_2/dense_2/MatMul/ReadVariableOp:b ^
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
&
_user_specified_namelstm_4_input
Ý
	
while_body_372806
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_5_split_readvariableop_resource_0:
B
3while_lstm_cell_5_split_1_readvariableop_resource_0:	?
+while_lstm_cell_5_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_5_split_readvariableop_resource:
@
1while_lstm_cell_5_split_1_readvariableop_resource:	=
)while_lstm_cell_5_readvariableop_resource:
˘ while/lstm_cell_5/ReadVariableOp˘"while/lstm_cell_5/ReadVariableOp_1˘"while/lstm_cell_5/ReadVariableOp_2˘"while/lstm_cell_5/ReadVariableOp_3˘&while/lstm_cell_5/split/ReadVariableOp˘(while/lstm_cell_5/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0d
!while/lstm_cell_5/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell_5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ž
while/lstm_cell_5/ones_likeFill*while/lstm_cell_5/ones_like/Shape:output:0*while/lstm_cell_5/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
while/lstm_cell_5/dropout/MulMul$while/lstm_cell_5/ones_like:output:0(while/lstm_cell_5/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
while/lstm_cell_5/dropout/ShapeShape$while/lstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:ą
6while/lstm_cell_5/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_5/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ý
&while/lstm_cell_5/dropout/GreaterEqualGreaterEqual?while/lstm_cell_5/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/dropout/CastCast*while/lstm_cell_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
while/lstm_cell_5/dropout/Mul_1Mul!while/lstm_cell_5/dropout/Mul:z:0"while/lstm_cell_5/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell_5/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ť
while/lstm_cell_5/dropout_1/MulMul$while/lstm_cell_5/ones_like:output:0*while/lstm_cell_5/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
!while/lstm_cell_5/dropout_1/ShapeShape$while/lstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:ľ
8while/lstm_cell_5/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_5/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0o
*while/lstm_cell_5/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ă
(while/lstm_cell_5/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_5/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_5/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_5/dropout_1/CastCast,while/lstm_cell_5/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
!while/lstm_cell_5/dropout_1/Mul_1Mul#while/lstm_cell_5/dropout_1/Mul:z:0$while/lstm_cell_5/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell_5/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ť
while/lstm_cell_5/dropout_2/MulMul$while/lstm_cell_5/ones_like:output:0*while/lstm_cell_5/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
!while/lstm_cell_5/dropout_2/ShapeShape$while/lstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:ľ
8while/lstm_cell_5/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_5/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0o
*while/lstm_cell_5/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ă
(while/lstm_cell_5/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_5/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_5/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_5/dropout_2/CastCast,while/lstm_cell_5/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
!while/lstm_cell_5/dropout_2/Mul_1Mul#while/lstm_cell_5/dropout_2/Mul:z:0$while/lstm_cell_5/dropout_2/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell_5/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ť
while/lstm_cell_5/dropout_3/MulMul$while/lstm_cell_5/ones_like:output:0*while/lstm_cell_5/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
!while/lstm_cell_5/dropout_3/ShapeShape$while/lstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:ľ
8while/lstm_cell_5/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_5/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0o
*while/lstm_cell_5/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ă
(while/lstm_cell_5/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_5/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_5/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_5/dropout_3/CastCast,while/lstm_cell_5/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
!while/lstm_cell_5/dropout_3/Mul_1Mul#while/lstm_cell_5/dropout_3/Mul:z:0$while/lstm_cell_5/dropout_3/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_5/split/ReadVariableOpReadVariableOp1while_lstm_cell_5_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ü
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0.while/lstm_cell_5/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_splitŠ
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_5/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_5/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_5/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_5/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
#while/lstm_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_5/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_5_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Î
while/lstm_cell_5/split_1Split,while/lstm_cell_5/split_1/split_dim:output:00while/lstm_cell_5/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell_5/BiasAddBiasAdd"while/lstm_cell_5/MatMul:product:0"while/lstm_cell_5/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_5/BiasAdd_1BiasAdd$while/lstm_cell_5/MatMul_1:product:0"while/lstm_cell_5/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_5/BiasAdd_2BiasAdd$while/lstm_cell_5/MatMul_2:product:0"while/lstm_cell_5/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_5/BiasAdd_3BiasAdd$while/lstm_cell_5/MatMul_3:product:0"while/lstm_cell_5/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mulMulwhile_placeholder_2#while/lstm_cell_5/dropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_1Mulwhile_placeholder_2%while/lstm_cell_5/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_2Mulwhile_placeholder_2%while/lstm_cell_5/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_3Mulwhile_placeholder_2%while/lstm_cell_5/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_5/ReadVariableOpReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell_5/strided_sliceStridedSlice(while/lstm_cell_5/ReadVariableOp:value:0.while/lstm_cell_5/strided_slice/stack:output:00while/lstm_cell_5/strided_slice/stack_1:output:00while/lstm_cell_5/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_5/MatMul_4MatMulwhile/lstm_cell_5/mul:z:0(while/lstm_cell_5/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/addAddV2"while/lstm_cell_5/BiasAdd:output:0$while/lstm_cell_5/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell_5/SigmoidSigmoidwhile/lstm_cell_5/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_5/ReadVariableOp_1ReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_5/strided_slice_1StridedSlice*while/lstm_cell_5/ReadVariableOp_1:value:00while/lstm_cell_5/strided_slice_1/stack:output:02while/lstm_cell_5/strided_slice_1/stack_1:output:02while/lstm_cell_5/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_5/MatMul_5MatMulwhile/lstm_cell_5/mul_1:z:0*while/lstm_cell_5/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_1AddV2$while/lstm_cell_5/BiasAdd_1:output:0$while/lstm_cell_5/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_5/Sigmoid_1Sigmoidwhile/lstm_cell_5/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_4Mulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_5/ReadVariableOp_2ReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_5/strided_slice_2StridedSlice*while/lstm_cell_5/ReadVariableOp_2:value:00while/lstm_cell_5/strided_slice_2/stack:output:02while/lstm_cell_5/strided_slice_2/stack_1:output:02while/lstm_cell_5/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_5/MatMul_6MatMulwhile/lstm_cell_5/mul_2:z:0*while/lstm_cell_5/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_2AddV2$while/lstm_cell_5/BiasAdd_2:output:0$while/lstm_cell_5/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
while/lstm_cell_5/TanhTanhwhile/lstm_cell_5/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_5Mulwhile/lstm_cell_5/Sigmoid:y:0while/lstm_cell_5/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_3AddV2while/lstm_cell_5/mul_4:z:0while/lstm_cell_5/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_5/ReadVariableOp_3ReadVariableOp+while_lstm_cell_5_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_5/strided_slice_3StridedSlice*while/lstm_cell_5/ReadVariableOp_3:value:00while/lstm_cell_5/strided_slice_3/stack:output:02while/lstm_cell_5/strided_slice_3/stack_1:output:02while/lstm_cell_5/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_5/MatMul_7MatMulwhile/lstm_cell_5/mul_3:z:0*while/lstm_cell_5/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/add_4AddV2$while/lstm_cell_5/BiasAdd_3:output:0$while/lstm_cell_5/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_5/Sigmoid_2Sigmoidwhile/lstm_cell_5/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/lstm_cell_5/Tanh_1Tanhwhile/lstm_cell_5/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_5/mul_6Mulwhile/lstm_cell_5/Sigmoid_2:y:0while/lstm_cell_5/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_6:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇy
while/Identity_4Identitywhile/lstm_cell_5/mul_6:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
while/Identity_5Identitywhile/lstm_cell_5/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˛

while/NoOpNoOp!^while/lstm_cell_5/ReadVariableOp#^while/lstm_cell_5/ReadVariableOp_1#^while/lstm_cell_5/ReadVariableOp_2#^while/lstm_cell_5/ReadVariableOp_3'^while/lstm_cell_5/split/ReadVariableOp)^while/lstm_cell_5/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_5_readvariableop_resource+while_lstm_cell_5_readvariableop_resource_0"h
1while_lstm_cell_5_split_1_readvariableop_resource3while_lstm_cell_5_split_1_readvariableop_resource_0"d
/while_lstm_cell_5_split_readvariableop_resource1while_lstm_cell_5_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2D
 while/lstm_cell_5/ReadVariableOp while/lstm_cell_5/ReadVariableOp2H
"while/lstm_cell_5/ReadVariableOp_1"while/lstm_cell_5/ReadVariableOp_12H
"while/lstm_cell_5/ReadVariableOp_2"while/lstm_cell_5/ReadVariableOp_22H
"while/lstm_cell_5/ReadVariableOp_3"while/lstm_cell_5/ReadVariableOp_32P
&while/lstm_cell_5/split/ReadVariableOp&while/lstm_cell_5/split/ReadVariableOp2T
(while/lstm_cell_5/split_1/ReadVariableOp(while/lstm_cell_5/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 
ý	
Ď
lstm_4_while_cond_373576*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3,
(lstm_4_while_less_lstm_4_strided_slice_1B
>lstm_4_while_lstm_4_while_cond_373576___redundant_placeholder0B
>lstm_4_while_lstm_4_while_cond_373576___redundant_placeholder1B
>lstm_4_while_lstm_4_while_cond_373576___redundant_placeholder2B
>lstm_4_while_lstm_4_while_cond_373576___redundant_placeholder3
lstm_4_while_identity
~
lstm_4/while/LessLesslstm_4_while_placeholder(lstm_4_while_less_lstm_4_strided_slice_1*
T0*
_output_shapes
: Y
lstm_4/while/IdentityIdentitylstm_4/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_4_while_identitylstm_4/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:
š
Ă
while_cond_374980
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_374980___redundant_placeholder04
0while_while_cond_374980___redundant_placeholder14
0while_while_cond_374980___redundant_placeholder24
0while_while_cond_374980___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:
{

lstm_4_while_body_373577*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3)
%lstm_4_while_lstm_4_strided_slice_1_0e
alstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0K
8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0:	9I
:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0:	F
2lstm_4_while_lstm_cell_4_readvariableop_resource_0:

lstm_4_while_identity
lstm_4_while_identity_1
lstm_4_while_identity_2
lstm_4_while_identity_3
lstm_4_while_identity_4
lstm_4_while_identity_5'
#lstm_4_while_lstm_4_strided_slice_1c
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensorI
6lstm_4_while_lstm_cell_4_split_readvariableop_resource:	9G
8lstm_4_while_lstm_cell_4_split_1_readvariableop_resource:	D
0lstm_4_while_lstm_cell_4_readvariableop_resource:
˘'lstm_4/while/lstm_cell_4/ReadVariableOp˘)lstm_4/while/lstm_cell_4/ReadVariableOp_1˘)lstm_4/while/lstm_cell_4/ReadVariableOp_2˘)lstm_4/while/lstm_cell_4/ReadVariableOp_3˘-lstm_4/while/lstm_cell_4/split/ReadVariableOp˘/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp
>lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   É
0lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0lstm_4_while_placeholderGlstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
element_dtype0r
(lstm_4/while/lstm_cell_4/ones_like/ShapeShapelstm_4_while_placeholder_2*
T0*
_output_shapes
:m
(lstm_4/while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ă
"lstm_4/while/lstm_cell_4/ones_likeFill1lstm_4/while/lstm_cell_4/ones_like/Shape:output:01lstm_4/while/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
(lstm_4/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :§
-lstm_4/while/lstm_cell_4/split/ReadVariableOpReadVariableOp8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	9*
dtype0í
lstm_4/while/lstm_cell_4/splitSplit1lstm_4/while/lstm_cell_4/split/split_dim:output:05lstm_4/while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_splitž
lstm_4/while/lstm_cell_4/MatMulMatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
!lstm_4/while/lstm_cell_4/MatMul_1MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
!lstm_4/while/lstm_cell_4/MatMul_2MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
!lstm_4/while/lstm_cell_4/MatMul_3MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l
*lstm_4/while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : §
/lstm_4/while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0ă
 lstm_4/while/lstm_cell_4/split_1Split3lstm_4/while/lstm_cell_4/split_1/split_dim:output:07lstm_4/while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split´
 lstm_4/while/lstm_cell_4/BiasAddBiasAdd)lstm_4/while/lstm_cell_4/MatMul:product:0)lstm_4/while/lstm_cell_4/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
"lstm_4/while/lstm_cell_4/BiasAdd_1BiasAdd+lstm_4/while/lstm_cell_4/MatMul_1:product:0)lstm_4/while/lstm_cell_4/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
"lstm_4/while/lstm_cell_4/BiasAdd_2BiasAdd+lstm_4/while/lstm_cell_4/MatMul_2:product:0)lstm_4/while/lstm_cell_4/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
"lstm_4/while/lstm_cell_4/BiasAdd_3BiasAdd+lstm_4/while/lstm_cell_4/MatMul_3:product:0)lstm_4/while/lstm_cell_4/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/while/lstm_cell_4/mulMullstm_4_while_placeholder_2+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
lstm_4/while/lstm_cell_4/mul_1Mullstm_4_while_placeholder_2+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
lstm_4/while/lstm_cell_4/mul_2Mullstm_4_while_placeholder_2+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
lstm_4/while/lstm_cell_4/mul_3Mullstm_4_while_placeholder_2+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
'lstm_4/while/lstm_cell_4/ReadVariableOpReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0}
,lstm_4/while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.lstm_4/while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.lstm_4/while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ę
&lstm_4/while/lstm_cell_4/strided_sliceStridedSlice/lstm_4/while/lstm_cell_4/ReadVariableOp:value:05lstm_4/while/lstm_cell_4/strided_slice/stack:output:07lstm_4/while/lstm_cell_4/strided_slice/stack_1:output:07lstm_4/while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maską
!lstm_4/while/lstm_cell_4/MatMul_4MatMul lstm_4/while/lstm_cell_4/mul:z:0/lstm_4/while/lstm_cell_4/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙°
lstm_4/while/lstm_cell_4/addAddV2)lstm_4/while/lstm_cell_4/BiasAdd:output:0+lstm_4/while/lstm_cell_4/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 lstm_4/while/lstm_cell_4/SigmoidSigmoid lstm_4/while/lstm_cell_4/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)lstm_4/while/lstm_cell_4/ReadVariableOp_1ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
.lstm_4/while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
0lstm_4/while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
0lstm_4/while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(lstm_4/while/lstm_cell_4/strided_slice_1StridedSlice1lstm_4/while/lstm_cell_4/ReadVariableOp_1:value:07lstm_4/while/lstm_cell_4/strided_slice_1/stack:output:09lstm_4/while/lstm_cell_4/strided_slice_1/stack_1:output:09lstm_4/while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskľ
!lstm_4/while/lstm_cell_4/MatMul_5MatMul"lstm_4/while/lstm_cell_4/mul_1:z:01lstm_4/while/lstm_cell_4/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙´
lstm_4/while/lstm_cell_4/add_1AddV2+lstm_4/while/lstm_cell_4/BiasAdd_1:output:0+lstm_4/while/lstm_cell_4/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"lstm_4/while/lstm_cell_4/Sigmoid_1Sigmoid"lstm_4/while/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/while/lstm_cell_4/mul_4Mul&lstm_4/while/lstm_cell_4/Sigmoid_1:y:0lstm_4_while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)lstm_4/while/lstm_cell_4/ReadVariableOp_2ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
.lstm_4/while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
0lstm_4/while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
0lstm_4/while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(lstm_4/while/lstm_cell_4/strided_slice_2StridedSlice1lstm_4/while/lstm_cell_4/ReadVariableOp_2:value:07lstm_4/while/lstm_cell_4/strided_slice_2/stack:output:09lstm_4/while/lstm_cell_4/strided_slice_2/stack_1:output:09lstm_4/while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskľ
!lstm_4/while/lstm_cell_4/MatMul_6MatMul"lstm_4/while/lstm_cell_4/mul_2:z:01lstm_4/while/lstm_cell_4/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙´
lstm_4/while/lstm_cell_4/add_2AddV2+lstm_4/while/lstm_cell_4/BiasAdd_2:output:0+lstm_4/while/lstm_cell_4/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_4/while/lstm_cell_4/TanhTanh"lstm_4/while/lstm_cell_4/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
lstm_4/while/lstm_cell_4/mul_5Mul$lstm_4/while/lstm_cell_4/Sigmoid:y:0!lstm_4/while/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
lstm_4/while/lstm_cell_4/add_3AddV2"lstm_4/while/lstm_cell_4/mul_4:z:0"lstm_4/while/lstm_cell_4/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)lstm_4/while/lstm_cell_4/ReadVariableOp_3ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
.lstm_4/while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
0lstm_4/while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0lstm_4/while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(lstm_4/while/lstm_cell_4/strided_slice_3StridedSlice1lstm_4/while/lstm_cell_4/ReadVariableOp_3:value:07lstm_4/while/lstm_cell_4/strided_slice_3/stack:output:09lstm_4/while/lstm_cell_4/strided_slice_3/stack_1:output:09lstm_4/while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskľ
!lstm_4/while/lstm_cell_4/MatMul_7MatMul"lstm_4/while/lstm_cell_4/mul_3:z:01lstm_4/while/lstm_cell_4/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙´
lstm_4/while/lstm_cell_4/add_4AddV2+lstm_4/while/lstm_cell_4/BiasAdd_3:output:0+lstm_4/while/lstm_cell_4/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"lstm_4/while/lstm_cell_4/Sigmoid_2Sigmoid"lstm_4/while/lstm_cell_4/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_4/while/lstm_cell_4/Tanh_1Tanh"lstm_4/while/lstm_cell_4/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ľ
lstm_4/while/lstm_cell_4/mul_6Mul&lstm_4/while/lstm_cell_4/Sigmoid_2:y:0#lstm_4/while/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ŕ
1lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_4_while_placeholder_1lstm_4_while_placeholder"lstm_4/while/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype0:éčŇT
lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_4/while/addAddV2lstm_4_while_placeholderlstm_4/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_4/while/add_1AddV2&lstm_4_while_lstm_4_while_loop_counterlstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_4/while/IdentityIdentitylstm_4/while/add_1:z:0^lstm_4/while/NoOp*
T0*
_output_shapes
: 
lstm_4/while/Identity_1Identity,lstm_4_while_lstm_4_while_maximum_iterations^lstm_4/while/NoOp*
T0*
_output_shapes
: n
lstm_4/while/Identity_2Identitylstm_4/while/add:z:0^lstm_4/while/NoOp*
T0*
_output_shapes
: Ž
lstm_4/while/Identity_3IdentityAlstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_4/while/NoOp*
T0*
_output_shapes
: :éčŇ
lstm_4/while/Identity_4Identity"lstm_4/while/lstm_cell_4/mul_6:z:0^lstm_4/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_4/while/Identity_5Identity"lstm_4/while/lstm_cell_4/add_3:z:0^lstm_4/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ă
lstm_4/while/NoOpNoOp(^lstm_4/while/lstm_cell_4/ReadVariableOp*^lstm_4/while/lstm_cell_4/ReadVariableOp_1*^lstm_4/while/lstm_cell_4/ReadVariableOp_2*^lstm_4/while/lstm_cell_4/ReadVariableOp_3.^lstm_4/while/lstm_cell_4/split/ReadVariableOp0^lstm_4/while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_4_while_identitylstm_4/while/Identity:output:0";
lstm_4_while_identity_1 lstm_4/while/Identity_1:output:0";
lstm_4_while_identity_2 lstm_4/while/Identity_2:output:0";
lstm_4_while_identity_3 lstm_4/while/Identity_3:output:0";
lstm_4_while_identity_4 lstm_4/while/Identity_4:output:0";
lstm_4_while_identity_5 lstm_4/while/Identity_5:output:0"L
#lstm_4_while_lstm_4_strided_slice_1%lstm_4_while_lstm_4_strided_slice_1_0"f
0lstm_4_while_lstm_cell_4_readvariableop_resource2lstm_4_while_lstm_cell_4_readvariableop_resource_0"v
8lstm_4_while_lstm_cell_4_split_1_readvariableop_resource:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0"r
6lstm_4_while_lstm_cell_4_split_readvariableop_resource8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0"Ä
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensoralstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2R
'lstm_4/while/lstm_cell_4/ReadVariableOp'lstm_4/while/lstm_cell_4/ReadVariableOp2V
)lstm_4/while/lstm_cell_4/ReadVariableOp_1)lstm_4/while/lstm_cell_4/ReadVariableOp_12V
)lstm_4/while/lstm_cell_4/ReadVariableOp_2)lstm_4/while/lstm_cell_4/ReadVariableOp_22V
)lstm_4/while/lstm_cell_4/ReadVariableOp_3)lstm_4/while/lstm_cell_4/ReadVariableOp_32^
-lstm_4/while/lstm_cell_4/split/ReadVariableOp-lstm_4/while/lstm_cell_4/split/ReadVariableOp2b
/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 
Ü
ß
B__inference_lstm_5_layer_call_and_return_conditional_losses_376750

inputs=
)lstm_cell_5_split_readvariableop_resource:
:
+lstm_cell_5_split_1_readvariableop_resource:	7
#lstm_cell_5_readvariableop_resource:

identity˘lstm_cell_5/ReadVariableOp˘lstm_cell_5/ReadVariableOp_1˘lstm_cell_5/ReadVariableOp_2˘lstm_cell_5/ReadVariableOp_3˘ lstm_cell_5/split/ReadVariableOp˘"lstm_cell_5/split_1/ReadVariableOp˘while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ę
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskY
lstm_cell_5/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell_5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_5/ones_likeFill$lstm_cell_5/ones_like/Shape:output:0$lstm_cell_5/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_5/dropout/MulMullstm_cell_5/ones_like:output:0"lstm_cell_5/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
lstm_cell_5/dropout/ShapeShapelstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:Ľ
0lstm_cell_5/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_5/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ë
 lstm_cell_5/dropout/GreaterEqualGreaterEqual9lstm_cell_5/dropout/random_uniform/RandomUniform:output:0+lstm_cell_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout/CastCast$lstm_cell_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout/Mul_1Mullstm_cell_5/dropout/Mul:z:0lstm_cell_5/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_5/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_5/dropout_1/MulMullstm_cell_5/ones_like:output:0$lstm_cell_5/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
lstm_cell_5/dropout_1/ShapeShapelstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:Š
2lstm_cell_5/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_5/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0i
$lstm_cell_5/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ń
"lstm_cell_5/dropout_1/GreaterEqualGreaterEqual;lstm_cell_5/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_5/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout_1/CastCast&lstm_cell_5/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout_1/Mul_1Mullstm_cell_5/dropout_1/Mul:z:0lstm_cell_5/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_5/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_5/dropout_2/MulMullstm_cell_5/ones_like:output:0$lstm_cell_5/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
lstm_cell_5/dropout_2/ShapeShapelstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:Š
2lstm_cell_5/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_5/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0i
$lstm_cell_5/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ń
"lstm_cell_5/dropout_2/GreaterEqualGreaterEqual;lstm_cell_5/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_5/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout_2/CastCast&lstm_cell_5/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout_2/Mul_1Mullstm_cell_5/dropout_2/Mul:z:0lstm_cell_5/dropout_2/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_5/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_5/dropout_3/MulMullstm_cell_5/ones_like:output:0$lstm_cell_5/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
lstm_cell_5/dropout_3/ShapeShapelstm_cell_5/ones_like:output:0*
T0*
_output_shapes
:Š
2lstm_cell_5/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_5/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0i
$lstm_cell_5/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ń
"lstm_cell_5/dropout_3/GreaterEqualGreaterEqual;lstm_cell_5/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_5/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout_3/CastCast&lstm_cell_5/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/dropout_3/Mul_1Mullstm_cell_5/dropout_3/Mul:z:0lstm_cell_5/dropout_3/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_5/split/ReadVariableOpReadVariableOp)lstm_cell_5_split_readvariableop_resource* 
_output_shapes
:
*
dtype0Ę
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0(lstm_cell_5/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0lstm_cell_5/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_5/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_5/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_5/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
lstm_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_5/split_1/ReadVariableOpReadVariableOp+lstm_cell_5_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ź
lstm_cell_5/split_1Split&lstm_cell_5/split_1/split_dim:output:0*lstm_cell_5/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell_5/BiasAddBiasAddlstm_cell_5/MatMul:product:0lstm_cell_5/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/BiasAdd_1BiasAddlstm_cell_5/MatMul_1:product:0lstm_cell_5/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/BiasAdd_2BiasAddlstm_cell_5/MatMul_2:product:0lstm_cell_5/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/BiasAdd_3BiasAddlstm_cell_5/MatMul_3:product:0lstm_cell_5/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_cell_5/mulMulzeros:output:0lstm_cell_5/dropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell_5/mul_1Mulzeros:output:0lstm_cell_5/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell_5/mul_2Mulzeros:output:0lstm_cell_5/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell_5/mul_3Mulzeros:output:0lstm_cell_5/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOpReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell_5/strided_sliceStridedSlice"lstm_cell_5/ReadVariableOp:value:0(lstm_cell_5/strided_slice/stack:output:0*lstm_cell_5/strided_slice/stack_1:output:0*lstm_cell_5/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_4MatMullstm_cell_5/mul:z:0"lstm_cell_5/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/addAddV2lstm_cell_5/BiasAdd:output:0lstm_cell_5/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell_5/SigmoidSigmoidlstm_cell_5/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOp_1ReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_5/strided_slice_1StridedSlice$lstm_cell_5/ReadVariableOp_1:value:0*lstm_cell_5/strided_slice_1/stack:output:0,lstm_cell_5/strided_slice_1/stack_1:output:0,lstm_cell_5/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_5MatMullstm_cell_5/mul_1:z:0$lstm_cell_5/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/add_1AddV2lstm_cell_5/BiasAdd_1:output:0lstm_cell_5/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
lstm_cell_5/mul_4Mullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOp_2ReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_5/strided_slice_2StridedSlice$lstm_cell_5/ReadVariableOp_2:value:0*lstm_cell_5/strided_slice_2/stack:output:0,lstm_cell_5/strided_slice_2/stack_1:output:0,lstm_cell_5/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_6MatMullstm_cell_5/mul_2:z:0$lstm_cell_5/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/add_2AddV2lstm_cell_5/BiasAdd_2:output:0lstm_cell_5/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
lstm_cell_5/TanhTanhlstm_cell_5/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
lstm_cell_5/mul_5Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
lstm_cell_5/add_3AddV2lstm_cell_5/mul_4:z:0lstm_cell_5/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/ReadVariableOp_3ReadVariableOp#lstm_cell_5_readvariableop_resource* 
_output_shapes
:
*
dtype0r
!lstm_cell_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ł
lstm_cell_5/strided_slice_3StridedSlice$lstm_cell_5/ReadVariableOp_3:value:0*lstm_cell_5/strided_slice_3/stack:output:0,lstm_cell_5/strided_slice_3/stack_1:output:0,lstm_cell_5/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_5/MatMul_7MatMullstm_cell_5/mul_3:z:0$lstm_cell_5/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_5/add_4AddV2lstm_cell_5/BiasAdd_3:output:0lstm_cell_5/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
lstm_cell_5/Tanh_1Tanhlstm_cell_5/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell_5/mul_6Mullstm_cell_5/Sigmoid_2:y:0lstm_cell_5/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ů
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_5_split_readvariableop_resource+lstm_cell_5_split_1_readvariableop_resource#lstm_cell_5_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_376591*
condR
while_cond_376590*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp^lstm_cell_5/ReadVariableOp^lstm_cell_5/ReadVariableOp_1^lstm_cell_5/ReadVariableOp_2^lstm_cell_5/ReadVariableOp_3!^lstm_cell_5/split/ReadVariableOp#^lstm_cell_5/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 28
lstm_cell_5/ReadVariableOplstm_cell_5/ReadVariableOp2<
lstm_cell_5/ReadVariableOp_1lstm_cell_5/ReadVariableOp_12<
lstm_cell_5/ReadVariableOp_2lstm_cell_5/ReadVariableOp_22<
lstm_cell_5/ReadVariableOp_3lstm_cell_5/ReadVariableOp_32D
 lstm_cell_5/split/ReadVariableOp lstm_cell_5/split/ReadVariableOp2H
"lstm_cell_5/split_1/ReadVariableOp"lstm_cell_5/split_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ő
	
while_body_374981
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_4_split_readvariableop_resource_0:	9B
3while_lstm_cell_4_split_1_readvariableop_resource_0:	?
+while_lstm_cell_4_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_4_split_readvariableop_resource:	9@
1while_lstm_cell_4_split_1_readvariableop_resource:	=
)while_lstm_cell_4_readvariableop_resource:
˘ while/lstm_cell_4/ReadVariableOp˘"while/lstm_cell_4/ReadVariableOp_1˘"while/lstm_cell_4/ReadVariableOp_2˘"while/lstm_cell_4/ReadVariableOp_3˘&while/lstm_cell_4/split/ReadVariableOp˘(while/lstm_cell_4/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
element_dtype0d
!while/lstm_cell_4/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ž
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
while/lstm_cell_4/dropout/MulMul$while/lstm_cell_4/ones_like:output:0(while/lstm_cell_4/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
while/lstm_cell_4/dropout/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:ą
6while/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_4/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ý
&while/lstm_cell_4/dropout/GreaterEqualGreaterEqual?while/lstm_cell_4/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/dropout/CastCast*while/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
while/lstm_cell_4/dropout/Mul_1Mul!while/lstm_cell_4/dropout/Mul:z:0"while/lstm_cell_4/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ť
while/lstm_cell_4/dropout_1/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
!while/lstm_cell_4/dropout_1/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:ľ
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0o
*while/lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ă
(while/lstm_cell_4/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_4/dropout_1/CastCast,while/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
!while/lstm_cell_4/dropout_1/Mul_1Mul#while/lstm_cell_4/dropout_1/Mul:z:0$while/lstm_cell_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ť
while/lstm_cell_4/dropout_2/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
!while/lstm_cell_4/dropout_2/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:ľ
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0o
*while/lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ă
(while/lstm_cell_4/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_4/dropout_2/CastCast,while/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
!while/lstm_cell_4/dropout_2/Mul_1Mul#while/lstm_cell_4/dropout_2/Mul:z:0$while/lstm_cell_4/dropout_2/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ť
while/lstm_cell_4/dropout_3/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
!while/lstm_cell_4/dropout_3/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:ľ
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0o
*while/lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ă
(while/lstm_cell_4/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_4/dropout_3/CastCast,while/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
!while/lstm_cell_4/dropout_3/Mul_1Mul#while/lstm_cell_4/dropout_3/Mul:z:0$while/lstm_cell_4/dropout_3/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	9*
dtype0Ř
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_splitŠ
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Î
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mulMulwhile_placeholder_2#while/lstm_cell_4/dropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_1Mulwhile_placeholder_2%while/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_2Mulwhile_placeholder_2%while/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_3Mulwhile_placeholder_2%while/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
while/lstm_cell_4/TanhTanhwhile/lstm_cell_4/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0while/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/lstm_cell_4/Tanh_1Tanhwhile/lstm_cell_4/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0while/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇy
while/Identity_4Identitywhile/lstm_cell_4/mul_6:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˛

while/NoOpNoOp!^while/lstm_cell_4/ReadVariableOp#^while/lstm_cell_4/ReadVariableOp_1#^while/lstm_cell_4/ReadVariableOp_2#^while/lstm_cell_4/ReadVariableOp_3'^while/lstm_cell_4/split/ReadVariableOp)^while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2D
 while/lstm_cell_4/ReadVariableOp while/lstm_cell_4/ReadVariableOp2H
"while/lstm_cell_4/ReadVariableOp_1"while/lstm_cell_4/ReadVariableOp_12H
"while/lstm_cell_4/ReadVariableOp_2"while/lstm_cell_4/ReadVariableOp_22H
"while/lstm_cell_4/ReadVariableOp_3"while/lstm_cell_4/ReadVariableOp_32P
&while/lstm_cell_4/split/ReadVariableOp&while/lstm_cell_4/split/ReadVariableOp2T
(while/lstm_cell_4/split_1/ReadVariableOp(while/lstm_cell_4/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 
÷
÷
,__inference_lstm_cell_5_layer_call_fn_377045

inputs
states_0
states_1
unknown:

	unknown_0:	
	unknown_1:

identity

identity_1

identity_2˘StatefulPartitionedCallŞ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_5_layer_call_and_return_conditional_losses_371713p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states/1
Ő
	
while_body_373121
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_4_split_readvariableop_resource_0:	9B
3while_lstm_cell_4_split_1_readvariableop_resource_0:	?
+while_lstm_cell_4_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_4_split_readvariableop_resource:	9@
1while_lstm_cell_4_split_1_readvariableop_resource:	=
)while_lstm_cell_4_readvariableop_resource:
˘ while/lstm_cell_4/ReadVariableOp˘"while/lstm_cell_4/ReadVariableOp_1˘"while/lstm_cell_4/ReadVariableOp_2˘"while/lstm_cell_4/ReadVariableOp_3˘&while/lstm_cell_4/split/ReadVariableOp˘(while/lstm_cell_4/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙9   Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9*
element_dtype0d
!while/lstm_cell_4/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ž
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
while/lstm_cell_4/dropout/MulMul$while/lstm_cell_4/ones_like:output:0(while/lstm_cell_4/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
while/lstm_cell_4/dropout/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:ą
6while/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_4/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ý
&while/lstm_cell_4/dropout/GreaterEqualGreaterEqual?while/lstm_cell_4/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/dropout/CastCast*while/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
while/lstm_cell_4/dropout/Mul_1Mul!while/lstm_cell_4/dropout/Mul:z:0"while/lstm_cell_4/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ť
while/lstm_cell_4/dropout_1/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
!while/lstm_cell_4/dropout_1/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:ľ
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0o
*while/lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ă
(while/lstm_cell_4/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_4/dropout_1/CastCast,while/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
!while/lstm_cell_4/dropout_1/Mul_1Mul#while/lstm_cell_4/dropout_1/Mul:z:0$while/lstm_cell_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ť
while/lstm_cell_4/dropout_2/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
!while/lstm_cell_4/dropout_2/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:ľ
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0o
*while/lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ă
(while/lstm_cell_4/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_4/dropout_2/CastCast,while/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
!while/lstm_cell_4/dropout_2/Mul_1Mul#while/lstm_cell_4/dropout_2/Mul:z:0$while/lstm_cell_4/dropout_2/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ť
while/lstm_cell_4/dropout_3/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
!while/lstm_cell_4/dropout_3/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:ľ
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0o
*while/lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>ă
(while/lstm_cell_4/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_4/dropout_3/CastCast,while/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
!while/lstm_cell_4/dropout_3/Mul_1Mul#while/lstm_cell_4/dropout_3/Mul:z:0$while/lstm_cell_4/dropout_3/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	9*
dtype0Ř
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_splitŠ
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Î
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mulMulwhile_placeholder_2#while/lstm_cell_4/dropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_1Mulwhile_placeholder_2%while/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_2Mulwhile_placeholder_2%while/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_3Mulwhile_placeholder_2%while/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
while/lstm_cell_4/TanhTanhwhile/lstm_cell_4/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0while/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0* 
_output_shapes
:
*
dtype0x
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ń
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask 
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/lstm_cell_4/Tanh_1Tanhwhile/lstm_cell_4/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0while/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇy
while/Identity_4Identitywhile/lstm_cell_4/mul_6:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˛

while/NoOpNoOp!^while/lstm_cell_4/ReadVariableOp#^while/lstm_cell_4/ReadVariableOp_1#^while/lstm_cell_4/ReadVariableOp_2#^while/lstm_cell_4/ReadVariableOp_3'^while/lstm_cell_4/split/ReadVariableOp)^while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2D
 while/lstm_cell_4/ReadVariableOp while/lstm_cell_4/ReadVariableOp2H
"while/lstm_cell_4/ReadVariableOp_1"while/lstm_cell_4/ReadVariableOp_12H
"while/lstm_cell_4/ReadVariableOp_2"while/lstm_cell_4/ReadVariableOp_22H
"while/lstm_cell_4/ReadVariableOp_3"while/lstm_cell_4/ReadVariableOp_32P
&while/lstm_cell_4/split/ReadVariableOp&while/lstm_cell_4/split/ReadVariableOp2T
(while/lstm_cell_4/split_1/ReadVariableOp(while/lstm_cell_4/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: 
Ş?
Ş
G__inference_lstm_cell_5_layer_call_and_return_conditional_losses_371713

inputs

states
states_11
split_readvariableop_resource:
.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘split/ReadVariableOp˘split_1/ReadVariableOpE
ones_like/ShapeShapestates*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
*
dtype0Ś
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
mulMulstatesones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
mul_1Mulstatesones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
mul_2Mulstatesones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
mul_3Mulstatesones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskf
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙X
mul_4MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mul_5MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
IdentityIdentity	mul_6:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_1Identity	mul_6:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namestates:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namestates
˛?
Ť
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_376921

inputs
states_0
states_10
split_readvariableop_resource:	9.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘split/ReadVariableOp˘split_1/ReadVariableOpG
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	9*
dtype0˘
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	9:	9:	9:	9*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
mulMulstates_0ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
mul_1Mulstates_0ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
mul_2Mulstates_0ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
mul_3Mulstates_0ones_like:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskf
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙X
mul_4MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mul_5MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
IdentityIdentity	mul_6:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_1Identity	mul_6:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙9:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙9
 
_user_specified_nameinputs:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states/1
Ł

ő
C__inference_dense_2_layer_call_and_return_conditional_losses_372096

inputs1
matmul_readvariableop_resource:	9-
biasadd_readvariableop_resource:9
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	9*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:9*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙9w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
š
Ă
while_cond_371258
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_371258___redundant_placeholder04
0while_while_cond_371258___redundant_placeholder14
0while_while_cond_371258___redundant_placeholder24
0while_while_cond_371258___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:
¤
ľ
'__inference_lstm_4_layer_call_fn_374607

inputs
unknown:	9
	unknown_0:	
	unknown_1:

identity˘StatefulPartitionedCallň
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_372388}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9: : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
 
_user_specified_nameinputs
8

B__inference_lstm_5_layer_call_and_return_conditional_losses_372062

inputs&
lstm_cell_5_371980:
!
lstm_cell_5_371982:	&
lstm_cell_5_371984:

identity˘#lstm_cell_5/StatefulPartitionedCall˘while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ę
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskó
#lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_5_371980lstm_cell_5_371982lstm_cell_5_371984*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_5_layer_call_and_return_conditional_losses_371934n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¸
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_5_371980lstm_cell_5_371982lstm_cell_5_371984*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_371993*
condR
while_cond_371992*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙t
NoOpNoOp$^lstm_cell_5/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 2J
#lstm_cell_5/StatefulPartitionedCall#lstm_cell_5/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ý	
Ď
lstm_5_while_cond_374371*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3,
(lstm_5_while_less_lstm_5_strided_slice_1B
>lstm_5_while_lstm_5_while_cond_374371___redundant_placeholder0B
>lstm_5_while_lstm_5_while_cond_374371___redundant_placeholder1B
>lstm_5_while_lstm_5_while_cond_374371___redundant_placeholder2B
>lstm_5_while_lstm_5_while_cond_374371___redundant_placeholder3
lstm_5_while_identity
~
lstm_5/while/LessLesslstm_5_while_placeholder(lstm_5_while_less_lstm_5_strided_slice_1*
T0*
_output_shapes
: Y
lstm_5/while/IdentityIdentitylstm_5/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_5_while_identitylstm_5/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:
š
Ă
while_cond_375807
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_375807___redundant_placeholder04
0while_while_cond_375807___redundant_placeholder14
0while_while_cond_375807___redundant_placeholder24
0while_while_cond_375807___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
:"ŰL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ů
serving_defaultĹ
R
lstm_4_inputB
serving_default_lstm_4_input:0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9S
time_distributed_2=
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9tensorflow/serving/predict:ÉŽ
Ű
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
Ú
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
Ú
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
°
	layer
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
ç
&iter

'beta_1

(beta_2
	)decay
*learning_rate+mt,mu-mv.mw/mx0my1mz2m{+v|,v}-v~.v/v0v1v2v"
	optimizer
X
+0
,1
-2
.3
/4
05
16
27"
trackable_list_wrapper
X
+0
,1
-2
.3
/4
05
16
27"
trackable_list_wrapper
 "
trackable_list_wrapper
Ę
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
2˙
-__inference_sequential_2_layer_call_fn_372659
-__inference_sequential_2_layer_call_fn_373454
-__inference_sequential_2_layer_call_fn_373475
-__inference_sequential_2_layer_call_fn_373377Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
î2ë
H__inference_sequential_2_layer_call_and_return_conditional_losses_373949
H__inference_sequential_2_layer_call_and_return_conditional_losses_374551
H__inference_sequential_2_layer_call_and_return_conditional_losses_373402
H__inference_sequential_2_layer_call_and_return_conditional_losses_373427Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ŃBÎ
!__inference__wrapped_model_371135lstm_4_input"
˛
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
,
8serving_default"
signature_map
ř
9
state_size

+kernel
,recurrent_kernel
-bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>_random_generator
?__call__
*@&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
š

Astates
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
˙2ü
'__inference_lstm_4_layer_call_fn_374585
'__inference_lstm_4_layer_call_fn_374596
'__inference_lstm_4_layer_call_fn_374607
'__inference_lstm_4_layer_call_fn_374618Ő
Ě˛Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ë2č
B__inference_lstm_4_layer_call_and_return_conditional_losses_374847
B__inference_lstm_4_layer_call_and_return_conditional_losses_375140
B__inference_lstm_4_layer_call_and_return_conditional_losses_375369
B__inference_lstm_4_layer_call_and_return_conditional_losses_375662Ő
Ě˛Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ř
G
state_size

.kernel
/recurrent_kernel
0bias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L_random_generator
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
š

Ostates
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
˙2ü
'__inference_lstm_5_layer_call_fn_375673
'__inference_lstm_5_layer_call_fn_375684
'__inference_lstm_5_layer_call_fn_375695
'__inference_lstm_5_layer_call_fn_375706Ő
Ě˛Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ë2č
B__inference_lstm_5_layer_call_and_return_conditional_losses_375935
B__inference_lstm_5_layer_call_and_return_conditional_losses_376228
B__inference_lstm_5_layer_call_and_return_conditional_losses_376457
B__inference_lstm_5_layer_call_and_return_conditional_losses_376750Ő
Ě˛Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ť

1kernel
2bias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_layer
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
­
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
°2­
3__inference_time_distributed_2_layer_call_fn_376759
3__inference_time_distributed_2_layer_call_fn_376768Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ć2ă
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_376790
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_376812Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
,:*	92lstm_4/lstm_cell_4/kernel
7:5
2#lstm_4/lstm_cell_4/recurrent_kernel
&:$2lstm_4/lstm_cell_4/bias
-:+
2lstm_5/lstm_cell_5/kernel
7:5
2#lstm_5/lstm_cell_5/recurrent_kernel
&:$2lstm_5/lstm_cell_5/bias
,:*	92time_distributed_2/kernel
%:#92time_distributed_2/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
`0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ĐBÍ
$__inference_signature_wrapper_374574lstm_4_input"
˛
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
:	variables
;trainable_variables
<regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 2
,__inference_lstm_cell_4_layer_call_fn_376829
,__inference_lstm_cell_4_layer_call_fn_376846ž
ľ˛ą
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ö2Ó
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_376921
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_377028ž
ľ˛ą
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
­
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 2
,__inference_lstm_cell_5_layer_call_fn_377045
,__inference_lstm_cell_5_layer_call_fn_377062ž
ľ˛ą
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ö2Ó
G__inference_lstm_cell_5_layer_call_and_return_conditional_losses_377137
G__inference_lstm_cell_5_layer_call_and_return_conditional_losses_377244ž
ľ˛ą
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
­
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
Ň2Ď
(__inference_dense_2_layer_call_fn_377253˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
í2ę
C__inference_dense_2_layer_call_and_return_conditional_losses_377264˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	ptotal
	qcount
r	variables
s	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
p0
q1"
trackable_list_wrapper
-
r	variables"
_generic_user_object
1:/	92 Adam/lstm_4/lstm_cell_4/kernel/m
<::
2*Adam/lstm_4/lstm_cell_4/recurrent_kernel/m
+:)2Adam/lstm_4/lstm_cell_4/bias/m
2:0
2 Adam/lstm_5/lstm_cell_5/kernel/m
<::
2*Adam/lstm_5/lstm_cell_5/recurrent_kernel/m
+:)2Adam/lstm_5/lstm_cell_5/bias/m
1:/	92 Adam/time_distributed_2/kernel/m
*:(92Adam/time_distributed_2/bias/m
1:/	92 Adam/lstm_4/lstm_cell_4/kernel/v
<::
2*Adam/lstm_4/lstm_cell_4/recurrent_kernel/v
+:)2Adam/lstm_4/lstm_cell_4/bias/v
2:0
2 Adam/lstm_5/lstm_cell_5/kernel/v
<::
2*Adam/lstm_5/lstm_cell_5/recurrent_kernel/v
+:)2Adam/lstm_5/lstm_cell_5/bias/v
1:/	92 Adam/time_distributed_2/kernel/v
*:(92Adam/time_distributed_2/bias/vĘ
!__inference__wrapped_model_371135¤+-,.0/12B˘?
8˘5
30
lstm_4_input˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
Ş "TŞQ
O
time_distributed_296
time_distributed_2˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9¤
C__inference_dense_2_layer_call_and_return_conditional_losses_377264]120˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙9
 |
(__inference_dense_2_layer_call_fn_377253P120˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙9Ň
B__inference_lstm_4_layer_call_and_return_conditional_losses_374847+-,O˘L
E˘B
41
/,
inputs/0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9

 
p 

 
Ş "3˘0
)&
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ň
B__inference_lstm_4_layer_call_and_return_conditional_losses_375140+-,O˘L
E˘B
41
/,
inputs/0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9

 
p

 
Ş "3˘0
)&
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ë
B__inference_lstm_4_layer_call_and_return_conditional_losses_375369+-,H˘E
>˘;
-*
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9

 
p 

 
Ş "3˘0
)&
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ë
B__inference_lstm_4_layer_call_and_return_conditional_losses_375662+-,H˘E
>˘;
-*
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9

 
p

 
Ş "3˘0
)&
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Š
'__inference_lstm_4_layer_call_fn_374585~+-,O˘L
E˘B
41
/,
inputs/0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9

 
p 

 
Ş "&#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Š
'__inference_lstm_4_layer_call_fn_374596~+-,O˘L
E˘B
41
/,
inputs/0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9

 
p

 
Ş "&#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˘
'__inference_lstm_4_layer_call_fn_374607w+-,H˘E
>˘;
-*
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9

 
p 

 
Ş "&#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˘
'__inference_lstm_4_layer_call_fn_374618w+-,H˘E
>˘;
-*
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9

 
p

 
Ş "&#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ó
B__inference_lstm_5_layer_call_and_return_conditional_losses_375935.0/P˘M
F˘C
52
0-
inputs/0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 
p 

 
Ş "3˘0
)&
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ó
B__inference_lstm_5_layer_call_and_return_conditional_losses_376228.0/P˘M
F˘C
52
0-
inputs/0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 
p

 
Ş "3˘0
)&
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ě
B__inference_lstm_5_layer_call_and_return_conditional_losses_376457.0/I˘F
?˘<
.+
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 
p 

 
Ş "3˘0
)&
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ě
B__inference_lstm_5_layer_call_and_return_conditional_losses_376750.0/I˘F
?˘<
.+
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 
p

 
Ş "3˘0
)&
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ş
'__inference_lstm_5_layer_call_fn_375673.0/P˘M
F˘C
52
0-
inputs/0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 
p 

 
Ş "&#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ş
'__inference_lstm_5_layer_call_fn_375684.0/P˘M
F˘C
52
0-
inputs/0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 
p

 
Ş "&#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ł
'__inference_lstm_5_layer_call_fn_375695x.0/I˘F
?˘<
.+
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 
p 

 
Ş "&#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ł
'__inference_lstm_5_layer_call_fn_375706x.0/I˘F
?˘<
.+
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 
p

 
Ş "&#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Î
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_376921+-,˘
x˘u
 
inputs˙˙˙˙˙˙˙˙˙9
M˘J
# 
states/0˙˙˙˙˙˙˙˙˙
# 
states/1˙˙˙˙˙˙˙˙˙
p 
Ş "v˘s
l˘i

0/0˙˙˙˙˙˙˙˙˙
GD
 
0/1/0˙˙˙˙˙˙˙˙˙
 
0/1/1˙˙˙˙˙˙˙˙˙
 Î
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_377028+-,˘
x˘u
 
inputs˙˙˙˙˙˙˙˙˙9
M˘J
# 
states/0˙˙˙˙˙˙˙˙˙
# 
states/1˙˙˙˙˙˙˙˙˙
p
Ş "v˘s
l˘i

0/0˙˙˙˙˙˙˙˙˙
GD
 
0/1/0˙˙˙˙˙˙˙˙˙
 
0/1/1˙˙˙˙˙˙˙˙˙
 Ł
,__inference_lstm_cell_4_layer_call_fn_376829ň+-,˘
x˘u
 
inputs˙˙˙˙˙˙˙˙˙9
M˘J
# 
states/0˙˙˙˙˙˙˙˙˙
# 
states/1˙˙˙˙˙˙˙˙˙
p 
Ş "f˘c

0˙˙˙˙˙˙˙˙˙
C@

1/0˙˙˙˙˙˙˙˙˙

1/1˙˙˙˙˙˙˙˙˙Ł
,__inference_lstm_cell_4_layer_call_fn_376846ň+-,˘
x˘u
 
inputs˙˙˙˙˙˙˙˙˙9
M˘J
# 
states/0˙˙˙˙˙˙˙˙˙
# 
states/1˙˙˙˙˙˙˙˙˙
p
Ş "f˘c

0˙˙˙˙˙˙˙˙˙
C@

1/0˙˙˙˙˙˙˙˙˙

1/1˙˙˙˙˙˙˙˙˙Đ
G__inference_lstm_cell_5_layer_call_and_return_conditional_losses_377137.0/˘
y˘v
!
inputs˙˙˙˙˙˙˙˙˙
M˘J
# 
states/0˙˙˙˙˙˙˙˙˙
# 
states/1˙˙˙˙˙˙˙˙˙
p 
Ş "v˘s
l˘i

0/0˙˙˙˙˙˙˙˙˙
GD
 
0/1/0˙˙˙˙˙˙˙˙˙
 
0/1/1˙˙˙˙˙˙˙˙˙
 Đ
G__inference_lstm_cell_5_layer_call_and_return_conditional_losses_377244.0/˘
y˘v
!
inputs˙˙˙˙˙˙˙˙˙
M˘J
# 
states/0˙˙˙˙˙˙˙˙˙
# 
states/1˙˙˙˙˙˙˙˙˙
p
Ş "v˘s
l˘i

0/0˙˙˙˙˙˙˙˙˙
GD
 
0/1/0˙˙˙˙˙˙˙˙˙
 
0/1/1˙˙˙˙˙˙˙˙˙
 Ľ
,__inference_lstm_cell_5_layer_call_fn_377045ô.0/˘
y˘v
!
inputs˙˙˙˙˙˙˙˙˙
M˘J
# 
states/0˙˙˙˙˙˙˙˙˙
# 
states/1˙˙˙˙˙˙˙˙˙
p 
Ş "f˘c

0˙˙˙˙˙˙˙˙˙
C@

1/0˙˙˙˙˙˙˙˙˙

1/1˙˙˙˙˙˙˙˙˙Ľ
,__inference_lstm_cell_5_layer_call_fn_377062ô.0/˘
y˘v
!
inputs˙˙˙˙˙˙˙˙˙
M˘J
# 
states/0˙˙˙˙˙˙˙˙˙
# 
states/1˙˙˙˙˙˙˙˙˙
p
Ş "f˘c

0˙˙˙˙˙˙˙˙˙
C@

1/0˙˙˙˙˙˙˙˙˙

1/1˙˙˙˙˙˙˙˙˙×
H__inference_sequential_2_layer_call_and_return_conditional_losses_373402+-,.0/12J˘G
@˘=
30
lstm_4_input˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
p 

 
Ş "2˘/
(%
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
 ×
H__inference_sequential_2_layer_call_and_return_conditional_losses_373427+-,.0/12J˘G
@˘=
30
lstm_4_input˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
p

 
Ş "2˘/
(%
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
 Ń
H__inference_sequential_2_layer_call_and_return_conditional_losses_373949+-,.0/12D˘A
:˘7
-*
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
p 

 
Ş "2˘/
(%
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
 Ń
H__inference_sequential_2_layer_call_and_return_conditional_losses_374551+-,.0/12D˘A
:˘7
-*
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
p

 
Ş "2˘/
(%
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
 Ž
-__inference_sequential_2_layer_call_fn_372659}+-,.0/12J˘G
@˘=
30
lstm_4_input˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
p 

 
Ş "%"˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9Ž
-__inference_sequential_2_layer_call_fn_373377}+-,.0/12J˘G
@˘=
30
lstm_4_input˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
p

 
Ş "%"˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9¨
-__inference_sequential_2_layer_call_fn_373454w+-,.0/12D˘A
:˘7
-*
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
p 

 
Ş "%"˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9¨
-__inference_sequential_2_layer_call_fn_373475w+-,.0/12D˘A
:˘7
-*
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
p

 
Ş "%"˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9Ý
$__inference_signature_wrapper_374574´+-,.0/12R˘O
˘ 
HŞE
C
lstm_4_input30
lstm_4_input˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9"TŞQ
O
time_distributed_296
time_distributed_2˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9Ń
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_37679012E˘B
;˘8
.+
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 

 
Ş "2˘/
(%
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
 Ń
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_37681212E˘B
;˘8
.+
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

 
Ş "2˘/
(%
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
 Š
3__inference_time_distributed_2_layer_call_fn_376759r12E˘B
;˘8
.+
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 

 
Ş "%"˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9Š
3__inference_time_distributed_2_layer_call_fn_376768r12E˘B
;˘8
.+
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

 
Ş "%"˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9