Ű
żŁ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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
ž
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02unknown8ŮÇ

tic_tac_toe_net/conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nametic_tac_toe_net/conv/kernel

/tic_tac_toe_net/conv/kernel/Read/ReadVariableOpReadVariableOptic_tac_toe_net/conv/kernel*'
_output_shapes
:*
dtype0

tic_tac_toe_net/conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nametic_tac_toe_net/conv/bias

-tic_tac_toe_net/conv/bias/Read/ReadVariableOpReadVariableOptic_tac_toe_net/conv/bias*
_output_shapes	
:*
dtype0

tic_tac_toe_net/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_nametic_tac_toe_net/dense/kernel

0tic_tac_toe_net/dense/kernel/Read/ReadVariableOpReadVariableOptic_tac_toe_net/dense/kernel* 
_output_shapes
:
*
dtype0

tic_tac_toe_net/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nametic_tac_toe_net/dense/bias

.tic_tac_toe_net/dense/bias/Read/ReadVariableOpReadVariableOptic_tac_toe_net/dense/bias*
_output_shapes	
:*
dtype0

tic_tac_toe_net/policy/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*.
shared_nametic_tac_toe_net/policy/kernel

1tic_tac_toe_net/policy/kernel/Read/ReadVariableOpReadVariableOptic_tac_toe_net/policy/kernel*
_output_shapes
:		*
dtype0

tic_tac_toe_net/policy/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_nametic_tac_toe_net/policy/bias

/tic_tac_toe_net/policy/bias/Read/ReadVariableOpReadVariableOptic_tac_toe_net/policy/bias*
_output_shapes
:	*
dtype0

tic_tac_toe_net/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_nametic_tac_toe_net/value/kernel

0tic_tac_toe_net/value/kernel/Read/ReadVariableOpReadVariableOptic_tac_toe_net/value/kernel*
_output_shapes
:	*
dtype0

tic_tac_toe_net/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nametic_tac_toe_net/value/bias

.tic_tac_toe_net/value/bias/Read/ReadVariableOpReadVariableOptic_tac_toe_net/value/bias*
_output_shapes
:*
dtype0
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
Š
"Adam/tic_tac_toe_net/conv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/tic_tac_toe_net/conv/kernel/m
˘
6Adam/tic_tac_toe_net/conv/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/tic_tac_toe_net/conv/kernel/m*'
_output_shapes
:*
dtype0

 Adam/tic_tac_toe_net/conv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/tic_tac_toe_net/conv/bias/m

4Adam/tic_tac_toe_net/conv/bias/m/Read/ReadVariableOpReadVariableOp Adam/tic_tac_toe_net/conv/bias/m*
_output_shapes	
:*
dtype0
¤
#Adam/tic_tac_toe_net/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#Adam/tic_tac_toe_net/dense/kernel/m

7Adam/tic_tac_toe_net/dense/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/tic_tac_toe_net/dense/kernel/m* 
_output_shapes
:
*
dtype0

!Adam/tic_tac_toe_net/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/tic_tac_toe_net/dense/bias/m

5Adam/tic_tac_toe_net/dense/bias/m/Read/ReadVariableOpReadVariableOp!Adam/tic_tac_toe_net/dense/bias/m*
_output_shapes	
:*
dtype0
Ľ
$Adam/tic_tac_toe_net/policy/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*5
shared_name&$Adam/tic_tac_toe_net/policy/kernel/m

8Adam/tic_tac_toe_net/policy/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/tic_tac_toe_net/policy/kernel/m*
_output_shapes
:		*
dtype0

"Adam/tic_tac_toe_net/policy/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/tic_tac_toe_net/policy/bias/m

6Adam/tic_tac_toe_net/policy/bias/m/Read/ReadVariableOpReadVariableOp"Adam/tic_tac_toe_net/policy/bias/m*
_output_shapes
:	*
dtype0
Ł
#Adam/tic_tac_toe_net/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*4
shared_name%#Adam/tic_tac_toe_net/value/kernel/m

7Adam/tic_tac_toe_net/value/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/tic_tac_toe_net/value/kernel/m*
_output_shapes
:	*
dtype0

!Adam/tic_tac_toe_net/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/tic_tac_toe_net/value/bias/m

5Adam/tic_tac_toe_net/value/bias/m/Read/ReadVariableOpReadVariableOp!Adam/tic_tac_toe_net/value/bias/m*
_output_shapes
:*
dtype0
Š
"Adam/tic_tac_toe_net/conv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/tic_tac_toe_net/conv/kernel/v
˘
6Adam/tic_tac_toe_net/conv/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/tic_tac_toe_net/conv/kernel/v*'
_output_shapes
:*
dtype0

 Adam/tic_tac_toe_net/conv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/tic_tac_toe_net/conv/bias/v

4Adam/tic_tac_toe_net/conv/bias/v/Read/ReadVariableOpReadVariableOp Adam/tic_tac_toe_net/conv/bias/v*
_output_shapes	
:*
dtype0
¤
#Adam/tic_tac_toe_net/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#Adam/tic_tac_toe_net/dense/kernel/v

7Adam/tic_tac_toe_net/dense/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/tic_tac_toe_net/dense/kernel/v* 
_output_shapes
:
*
dtype0

!Adam/tic_tac_toe_net/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/tic_tac_toe_net/dense/bias/v

5Adam/tic_tac_toe_net/dense/bias/v/Read/ReadVariableOpReadVariableOp!Adam/tic_tac_toe_net/dense/bias/v*
_output_shapes	
:*
dtype0
Ľ
$Adam/tic_tac_toe_net/policy/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*5
shared_name&$Adam/tic_tac_toe_net/policy/kernel/v

8Adam/tic_tac_toe_net/policy/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/tic_tac_toe_net/policy/kernel/v*
_output_shapes
:		*
dtype0

"Adam/tic_tac_toe_net/policy/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/tic_tac_toe_net/policy/bias/v

6Adam/tic_tac_toe_net/policy/bias/v/Read/ReadVariableOpReadVariableOp"Adam/tic_tac_toe_net/policy/bias/v*
_output_shapes
:	*
dtype0
Ł
#Adam/tic_tac_toe_net/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*4
shared_name%#Adam/tic_tac_toe_net/value/kernel/v

7Adam/tic_tac_toe_net/value/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/tic_tac_toe_net/value/kernel/v*
_output_shapes
:	*
dtype0

!Adam/tic_tac_toe_net/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/tic_tac_toe_net/value/bias/v

5Adam/tic_tac_toe_net/value/bias/v/Read/ReadVariableOpReadVariableOp!Adam/tic_tac_toe_net/value/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
1
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ż0
valueľ0B˛0 BŤ0
´
conv
flatten
	dense

policy
	value
	optimizer
loss
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
h

#kernel
$bias
%trainable_variables
&regularization_losses
'	variables
(	keras_api
Đ
)iter

*beta_1

+beta_2
	,decay
-learning_ratem[m\m]m^m_m`#ma$mbvcvdvevfvgvh#vi$vj
 
8
0
1
2
3
4
5
#6
$7
 
8
0
1
2
3
4
5
#6
$7
­
.layer_regularization_losses
/layer_metrics
0non_trainable_variables

1layers
trainable_variables
	regularization_losses

	variables
2metrics
 
WU
VARIABLE_VALUEtic_tac_toe_net/conv/kernel&conv/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEtic_tac_toe_net/conv/bias$conv/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
3layer_metrics
4non_trainable_variables

5layers
trainable_variables
regularization_losses
6layer_regularization_losses
	variables
7metrics
 
 
 
­
8layer_metrics
9non_trainable_variables

:layers
trainable_variables
regularization_losses
;layer_regularization_losses
	variables
<metrics
YW
VARIABLE_VALUEtic_tac_toe_net/dense/kernel'dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtic_tac_toe_net/dense/bias%dense/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
=layer_metrics
>non_trainable_variables

?layers
trainable_variables
regularization_losses
@layer_regularization_losses
	variables
Ametrics
[Y
VARIABLE_VALUEtic_tac_toe_net/policy/kernel(policy/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtic_tac_toe_net/policy/bias&policy/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
Blayer_metrics
Cnon_trainable_variables

Dlayers
trainable_variables
 regularization_losses
Elayer_regularization_losses
!	variables
Fmetrics
YW
VARIABLE_VALUEtic_tac_toe_net/value/kernel'value/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtic_tac_toe_net/value/bias%value/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1
 

#0
$1
­
Glayer_metrics
Hnon_trainable_variables

Ilayers
%trainable_variables
&regularization_losses
Jlayer_regularization_losses
'	variables
Kmetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
#
0
1
2
3
4

L0
M1
N2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ototal
	Pcount
Q	variables
R	keras_api
4
	Stotal
	Tcount
U	variables
V	keras_api
4
	Wtotal
	Xcount
Y	variables
Z	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

O0
P1

Q	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

S0
T1

U	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

W0
X1

Y	variables
zx
VARIABLE_VALUE"Adam/tic_tac_toe_net/conv/kernel/mBconv/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE Adam/tic_tac_toe_net/conv/bias/m@conv/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/tic_tac_toe_net/dense/kernel/mCdense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE!Adam/tic_tac_toe_net/dense/bias/mAdense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE$Adam/tic_tac_toe_net/policy/kernel/mDpolicy/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/tic_tac_toe_net/policy/bias/mBpolicy/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/tic_tac_toe_net/value/kernel/mCvalue/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE!Adam/tic_tac_toe_net/value/bias/mAvalue/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/tic_tac_toe_net/conv/kernel/vBconv/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE Adam/tic_tac_toe_net/conv/bias/v@conv/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/tic_tac_toe_net/dense/kernel/vCdense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE!Adam/tic_tac_toe_net/dense/bias/vAdense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE$Adam/tic_tac_toe_net/policy/kernel/vDpolicy/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/tic_tac_toe_net/policy/bias/vBpolicy/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/tic_tac_toe_net/value/kernel/vCvalue/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE!Adam/tic_tac_toe_net/value/bias/vAvalue/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*$
shape:˙˙˙˙˙˙˙˙˙
Â
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1tic_tac_toe_net/conv/kerneltic_tac_toe_net/conv/biastic_tac_toe_net/dense/kerneltic_tac_toe_net/dense/biastic_tac_toe_net/policy/kerneltic_tac_toe_net/policy/biastic_tac_toe_net/value/kerneltic_tac_toe_net/value/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙	:˙˙˙˙˙˙˙˙˙**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *1
f,R*
(__inference_signature_wrapper_1973734706
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/tic_tac_toe_net/conv/kernel/Read/ReadVariableOp-tic_tac_toe_net/conv/bias/Read/ReadVariableOp0tic_tac_toe_net/dense/kernel/Read/ReadVariableOp.tic_tac_toe_net/dense/bias/Read/ReadVariableOp1tic_tac_toe_net/policy/kernel/Read/ReadVariableOp/tic_tac_toe_net/policy/bias/Read/ReadVariableOp0tic_tac_toe_net/value/kernel/Read/ReadVariableOp.tic_tac_toe_net/value/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp6Adam/tic_tac_toe_net/conv/kernel/m/Read/ReadVariableOp4Adam/tic_tac_toe_net/conv/bias/m/Read/ReadVariableOp7Adam/tic_tac_toe_net/dense/kernel/m/Read/ReadVariableOp5Adam/tic_tac_toe_net/dense/bias/m/Read/ReadVariableOp8Adam/tic_tac_toe_net/policy/kernel/m/Read/ReadVariableOp6Adam/tic_tac_toe_net/policy/bias/m/Read/ReadVariableOp7Adam/tic_tac_toe_net/value/kernel/m/Read/ReadVariableOp5Adam/tic_tac_toe_net/value/bias/m/Read/ReadVariableOp6Adam/tic_tac_toe_net/conv/kernel/v/Read/ReadVariableOp4Adam/tic_tac_toe_net/conv/bias/v/Read/ReadVariableOp7Adam/tic_tac_toe_net/dense/kernel/v/Read/ReadVariableOp5Adam/tic_tac_toe_net/dense/bias/v/Read/ReadVariableOp8Adam/tic_tac_toe_net/policy/kernel/v/Read/ReadVariableOp6Adam/tic_tac_toe_net/policy/bias/v/Read/ReadVariableOp7Adam/tic_tac_toe_net/value/kernel/v/Read/ReadVariableOp5Adam/tic_tac_toe_net/value/bias/v/Read/ReadVariableOpConst*0
Tin)
'2%	*
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
GPU 2J 8 *,
f'R%
#__inference__traced_save_1973734926
Ü	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametic_tac_toe_net/conv/kerneltic_tac_toe_net/conv/biastic_tac_toe_net/dense/kerneltic_tac_toe_net/dense/biastic_tac_toe_net/policy/kerneltic_tac_toe_net/policy/biastic_tac_toe_net/value/kerneltic_tac_toe_net/value/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2"Adam/tic_tac_toe_net/conv/kernel/m Adam/tic_tac_toe_net/conv/bias/m#Adam/tic_tac_toe_net/dense/kernel/m!Adam/tic_tac_toe_net/dense/bias/m$Adam/tic_tac_toe_net/policy/kernel/m"Adam/tic_tac_toe_net/policy/bias/m#Adam/tic_tac_toe_net/value/kernel/m!Adam/tic_tac_toe_net/value/bias/m"Adam/tic_tac_toe_net/conv/kernel/v Adam/tic_tac_toe_net/conv/bias/v#Adam/tic_tac_toe_net/dense/kernel/v!Adam/tic_tac_toe_net/dense/bias/v$Adam/tic_tac_toe_net/policy/kernel/v"Adam/tic_tac_toe_net/policy/bias/v#Adam/tic_tac_toe_net/value/kernel/v!Adam/tic_tac_toe_net/value/bias/v*/
Tin(
&2$*
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
GPU 2J 8 */
f*R(
&__inference__traced_restore_1973735041úŤ
+
Š
%__inference__wrapped_model_1973734521
input_17
3tic_tac_toe_net_conv_conv2d_readvariableop_resource8
4tic_tac_toe_net_conv_biasadd_readvariableop_resource8
4tic_tac_toe_net_dense_matmul_readvariableop_resource9
5tic_tac_toe_net_dense_biasadd_readvariableop_resource9
5tic_tac_toe_net_policy_matmul_readvariableop_resource:
6tic_tac_toe_net_policy_biasadd_readvariableop_resource8
4tic_tac_toe_net_value_matmul_readvariableop_resource9
5tic_tac_toe_net_value_biasadd_readvariableop_resource
identity

identity_1Ő
*tic_tac_toe_net/conv/Conv2D/ReadVariableOpReadVariableOp3tic_tac_toe_net_conv_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02,
*tic_tac_toe_net/conv/Conv2D/ReadVariableOpĺ
tic_tac_toe_net/conv/Conv2DConv2Dinput_12tic_tac_toe_net/conv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
tic_tac_toe_net/conv/Conv2DĚ
+tic_tac_toe_net/conv/BiasAdd/ReadVariableOpReadVariableOp4tic_tac_toe_net_conv_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+tic_tac_toe_net/conv/BiasAdd/ReadVariableOpÝ
tic_tac_toe_net/conv/BiasAddBiasAdd$tic_tac_toe_net/conv/Conv2D:output:03tic_tac_toe_net/conv/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
tic_tac_toe_net/conv/BiasAdd 
tic_tac_toe_net/conv/ReluRelu%tic_tac_toe_net/conv/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
tic_tac_toe_net/conv/Relu
tic_tac_toe_net/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   2
tic_tac_toe_net/flatten/ConstŃ
tic_tac_toe_net/flatten/ReshapeReshape'tic_tac_toe_net/conv/Relu:activations:0&tic_tac_toe_net/flatten/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
tic_tac_toe_net/flatten/ReshapeŃ
+tic_tac_toe_net/dense/MatMul/ReadVariableOpReadVariableOp4tic_tac_toe_net_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02-
+tic_tac_toe_net/dense/MatMul/ReadVariableOpŘ
tic_tac_toe_net/dense/MatMulMatMul(tic_tac_toe_net/flatten/Reshape:output:03tic_tac_toe_net/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
tic_tac_toe_net/dense/MatMulĎ
,tic_tac_toe_net/dense/BiasAdd/ReadVariableOpReadVariableOp5tic_tac_toe_net_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,tic_tac_toe_net/dense/BiasAdd/ReadVariableOpÚ
tic_tac_toe_net/dense/BiasAddBiasAdd&tic_tac_toe_net/dense/MatMul:product:04tic_tac_toe_net/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
tic_tac_toe_net/dense/BiasAdd
tic_tac_toe_net/dense/ReluRelu&tic_tac_toe_net/dense/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
tic_tac_toe_net/dense/ReluÓ
,tic_tac_toe_net/policy/MatMul/ReadVariableOpReadVariableOp5tic_tac_toe_net_policy_matmul_readvariableop_resource*
_output_shapes
:		*
dtype02.
,tic_tac_toe_net/policy/MatMul/ReadVariableOpÚ
tic_tac_toe_net/policy/MatMulMatMul(tic_tac_toe_net/dense/Relu:activations:04tic_tac_toe_net/policy/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙	2
tic_tac_toe_net/policy/MatMulŃ
-tic_tac_toe_net/policy/BiasAdd/ReadVariableOpReadVariableOp6tic_tac_toe_net_policy_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02/
-tic_tac_toe_net/policy/BiasAdd/ReadVariableOpÝ
tic_tac_toe_net/policy/BiasAddBiasAdd'tic_tac_toe_net/policy/MatMul:product:05tic_tac_toe_net/policy/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙	2 
tic_tac_toe_net/policy/BiasAddŚ
tic_tac_toe_net/policy/SoftmaxSoftmax'tic_tac_toe_net/policy/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙	2 
tic_tac_toe_net/policy/SoftmaxĐ
+tic_tac_toe_net/value/MatMul/ReadVariableOpReadVariableOp4tic_tac_toe_net_value_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+tic_tac_toe_net/value/MatMul/ReadVariableOp×
tic_tac_toe_net/value/MatMulMatMul(tic_tac_toe_net/dense/Relu:activations:03tic_tac_toe_net/value/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
tic_tac_toe_net/value/MatMulÎ
,tic_tac_toe_net/value/BiasAdd/ReadVariableOpReadVariableOp5tic_tac_toe_net_value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,tic_tac_toe_net/value/BiasAdd/ReadVariableOpŮ
tic_tac_toe_net/value/BiasAddBiasAdd&tic_tac_toe_net/value/MatMul:product:04tic_tac_toe_net/value/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
tic_tac_toe_net/value/BiasAdd
tic_tac_toe_net/value/TanhTanh&tic_tac_toe_net/value/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
tic_tac_toe_net/value/Tanh|
IdentityIdentity(tic_tac_toe_net/policy/Softmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙	2

Identityv

Identity_1Identitytic_tac_toe_net/value/Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*N
_input_shapes=
;:˙˙˙˙˙˙˙˙˙:::::::::X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
ł
­
E__inference_dense_layer_call_and_return_conditional_losses_1973734748

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
M
Ě
#__inference__traced_save_1973734926
file_prefix:
6savev2_tic_tac_toe_net_conv_kernel_read_readvariableop8
4savev2_tic_tac_toe_net_conv_bias_read_readvariableop;
7savev2_tic_tac_toe_net_dense_kernel_read_readvariableop9
5savev2_tic_tac_toe_net_dense_bias_read_readvariableop<
8savev2_tic_tac_toe_net_policy_kernel_read_readvariableop:
6savev2_tic_tac_toe_net_policy_bias_read_readvariableop;
7savev2_tic_tac_toe_net_value_kernel_read_readvariableop9
5savev2_tic_tac_toe_net_value_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableopA
=savev2_adam_tic_tac_toe_net_conv_kernel_m_read_readvariableop?
;savev2_adam_tic_tac_toe_net_conv_bias_m_read_readvariableopB
>savev2_adam_tic_tac_toe_net_dense_kernel_m_read_readvariableop@
<savev2_adam_tic_tac_toe_net_dense_bias_m_read_readvariableopC
?savev2_adam_tic_tac_toe_net_policy_kernel_m_read_readvariableopA
=savev2_adam_tic_tac_toe_net_policy_bias_m_read_readvariableopB
>savev2_adam_tic_tac_toe_net_value_kernel_m_read_readvariableop@
<savev2_adam_tic_tac_toe_net_value_bias_m_read_readvariableopA
=savev2_adam_tic_tac_toe_net_conv_kernel_v_read_readvariableop?
;savev2_adam_tic_tac_toe_net_conv_bias_v_read_readvariableopB
>savev2_adam_tic_tac_toe_net_dense_kernel_v_read_readvariableop@
<savev2_adam_tic_tac_toe_net_dense_bias_v_read_readvariableopC
?savev2_adam_tic_tac_toe_net_policy_kernel_v_read_readvariableopA
=savev2_adam_tic_tac_toe_net_policy_bias_v_read_readvariableopB
>savev2_adam_tic_tac_toe_net_value_kernel_v_read_readvariableop@
<savev2_adam_tic_tac_toe_net_value_bias_v_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_037691078a284fdaba7bbdb97a3caa80/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameĘ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*Ü
valueŇBĎ$B&conv/kernel/.ATTRIBUTES/VARIABLE_VALUEB$conv/bias/.ATTRIBUTES/VARIABLE_VALUEB'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB(policy/kernel/.ATTRIBUTES/VARIABLE_VALUEB&policy/bias/.ATTRIBUTES/VARIABLE_VALUEB'value/kernel/.ATTRIBUTES/VARIABLE_VALUEB%value/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBBconv/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@conv/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCdense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAdense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDpolicy/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBpolicy/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvalue/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAvalue/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBconv/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@conv/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCdense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAdense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDpolicy/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBpolicy/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvalue/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAvalue/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesĐ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices­
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_tic_tac_toe_net_conv_kernel_read_readvariableop4savev2_tic_tac_toe_net_conv_bias_read_readvariableop7savev2_tic_tac_toe_net_dense_kernel_read_readvariableop5savev2_tic_tac_toe_net_dense_bias_read_readvariableop8savev2_tic_tac_toe_net_policy_kernel_read_readvariableop6savev2_tic_tac_toe_net_policy_bias_read_readvariableop7savev2_tic_tac_toe_net_value_kernel_read_readvariableop5savev2_tic_tac_toe_net_value_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop=savev2_adam_tic_tac_toe_net_conv_kernel_m_read_readvariableop;savev2_adam_tic_tac_toe_net_conv_bias_m_read_readvariableop>savev2_adam_tic_tac_toe_net_dense_kernel_m_read_readvariableop<savev2_adam_tic_tac_toe_net_dense_bias_m_read_readvariableop?savev2_adam_tic_tac_toe_net_policy_kernel_m_read_readvariableop=savev2_adam_tic_tac_toe_net_policy_bias_m_read_readvariableop>savev2_adam_tic_tac_toe_net_value_kernel_m_read_readvariableop<savev2_adam_tic_tac_toe_net_value_bias_m_read_readvariableop=savev2_adam_tic_tac_toe_net_conv_kernel_v_read_readvariableop;savev2_adam_tic_tac_toe_net_conv_bias_v_read_readvariableop>savev2_adam_tic_tac_toe_net_dense_kernel_v_read_readvariableop<savev2_adam_tic_tac_toe_net_dense_bias_v_read_readvariableop?savev2_adam_tic_tac_toe_net_policy_kernel_v_read_readvariableop=savev2_adam_tic_tac_toe_net_policy_bias_v_read_readvariableop>savev2_adam_tic_tac_toe_net_value_kernel_v_read_readvariableop<savev2_adam_tic_tac_toe_net_value_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$	2
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĄ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: :::
::		:	:	:: : : : : : : : : : : :::
::		:	:	::::
::		:	:	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
::!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:		: 

_output_shapes
:	:%!

_output_shapes
:	: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
::!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:		: 

_output_shapes
:	:%!

_output_shapes
:	: 

_output_shapes
::-)
'
_output_shapes
::!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::% !

_output_shapes
:		: !

_output_shapes
:	:%"!

_output_shapes
:	: #

_output_shapes
::$

_output_shapes
: 
ă

+__inference_policy_layer_call_fn_1973734777

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_policy_layer_call_and_return_conditional_losses_19737346042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙	2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ł
­
E__inference_value_layer_call_and_return_conditional_losses_1973734631

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
Ź
D__inference_conv_layer_call_and_return_conditional_losses_1973734717

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpĽ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙:::W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ţ
~
)__inference_conv_layer_call_fn_1973734726

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv_layer_call_and_return_conditional_losses_19737345362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ě	
č
(__inference_signature_wrapper_1973734706
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1˘StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙	:˙˙˙˙˙˙˙˙˙**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference__wrapped_model_19737345212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙	2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*N
_input_shapes=
;:˙˙˙˙˙˙˙˙˙::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
Á
c
G__inference_flatten_layer_call_and_return_conditional_losses_1973734732

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ę
Ż
O__inference_tic_tac_toe_net_layer_call_and_return_conditional_losses_1973734649
input_1
conv_1973734547
conv_1973734549
dense_1973734588
dense_1973734590
policy_1973734615
policy_1973734617
value_1973734642
value_1973734644
identity

identity_1˘conv/StatefulPartitionedCall˘dense/StatefulPartitionedCall˘policy/StatefulPartitionedCall˘value/StatefulPartitionedCall
conv/StatefulPartitionedCallStatefulPartitionedCallinput_1conv_1973734547conv_1973734549*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv_layer_call_and_return_conditional_losses_19737345362
conv/StatefulPartitionedCallő
flatten/PartitionedCallPartitionedCall%conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_layer_call_and_return_conditional_losses_19737345582
flatten/PartitionedCallŹ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1973734588dense_1973734590*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_layer_call_and_return_conditional_losses_19737345772
dense/StatefulPartitionedCallś
policy/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0policy_1973734615policy_1973734617*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_policy_layer_call_and_return_conditional_losses_19737346042 
policy/StatefulPartitionedCallą
value/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0value_1973734642value_1973734644*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_value_layer_call_and_return_conditional_losses_19737346312
value/StatefulPartitionedCallű
IdentityIdentity'policy/StatefulPartitionedCall:output:0^conv/StatefulPartitionedCall^dense/StatefulPartitionedCall^policy/StatefulPartitionedCall^value/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙	2

Identityţ

Identity_1Identity&value/StatefulPartitionedCall:output:0^conv/StatefulPartitionedCall^dense/StatefulPartitionedCall^policy/StatefulPartitionedCall^value/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*N
_input_shapes=
;:˙˙˙˙˙˙˙˙˙::::::::2<
conv/StatefulPartitionedCallconv/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
policy/StatefulPartitionedCallpolicy/StatefulPartitionedCall2>
value/StatefulPartitionedCallvalue/StatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
ś
Ž
F__inference_policy_layer_call_and_return_conditional_losses_1973734768

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:		*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙	2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙	2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙	2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙	2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ł
­
E__inference_value_layer_call_and_return_conditional_losses_1973734788

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
â

*__inference_dense_layer_call_fn_1973734757

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_layer_call_and_return_conditional_losses_19737345772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ź
H
,__inference_flatten_layer_call_fn_1973734737

inputs
identityĆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_layer_call_and_return_conditional_losses_19737345582
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ś

&__inference__traced_restore_1973735041
file_prefix0
,assignvariableop_tic_tac_toe_net_conv_kernel0
,assignvariableop_1_tic_tac_toe_net_conv_bias3
/assignvariableop_2_tic_tac_toe_net_dense_kernel1
-assignvariableop_3_tic_tac_toe_net_dense_bias4
0assignvariableop_4_tic_tac_toe_net_policy_kernel2
.assignvariableop_5_tic_tac_toe_net_policy_bias3
/assignvariableop_6_tic_tac_toe_net_value_kernel1
-assignvariableop_7_tic_tac_toe_net_value_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1
assignvariableop_17_total_2
assignvariableop_18_count_2:
6assignvariableop_19_adam_tic_tac_toe_net_conv_kernel_m8
4assignvariableop_20_adam_tic_tac_toe_net_conv_bias_m;
7assignvariableop_21_adam_tic_tac_toe_net_dense_kernel_m9
5assignvariableop_22_adam_tic_tac_toe_net_dense_bias_m<
8assignvariableop_23_adam_tic_tac_toe_net_policy_kernel_m:
6assignvariableop_24_adam_tic_tac_toe_net_policy_bias_m;
7assignvariableop_25_adam_tic_tac_toe_net_value_kernel_m9
5assignvariableop_26_adam_tic_tac_toe_net_value_bias_m:
6assignvariableop_27_adam_tic_tac_toe_net_conv_kernel_v8
4assignvariableop_28_adam_tic_tac_toe_net_conv_bias_v;
7assignvariableop_29_adam_tic_tac_toe_net_dense_kernel_v9
5assignvariableop_30_adam_tic_tac_toe_net_dense_bias_v<
8assignvariableop_31_adam_tic_tac_toe_net_policy_kernel_v:
6assignvariableop_32_adam_tic_tac_toe_net_policy_bias_v;
7assignvariableop_33_adam_tic_tac_toe_net_value_kernel_v9
5assignvariableop_34_adam_tic_tac_toe_net_value_bias_v
identity_36˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_18˘AssignVariableOp_19˘AssignVariableOp_2˘AssignVariableOp_20˘AssignVariableOp_21˘AssignVariableOp_22˘AssignVariableOp_23˘AssignVariableOp_24˘AssignVariableOp_25˘AssignVariableOp_26˘AssignVariableOp_27˘AssignVariableOp_28˘AssignVariableOp_29˘AssignVariableOp_3˘AssignVariableOp_30˘AssignVariableOp_31˘AssignVariableOp_32˘AssignVariableOp_33˘AssignVariableOp_34˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9Đ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*Ü
valueŇBĎ$B&conv/kernel/.ATTRIBUTES/VARIABLE_VALUEB$conv/bias/.ATTRIBUTES/VARIABLE_VALUEB'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB(policy/kernel/.ATTRIBUTES/VARIABLE_VALUEB&policy/bias/.ATTRIBUTES/VARIABLE_VALUEB'value/kernel/.ATTRIBUTES/VARIABLE_VALUEB%value/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBBconv/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@conv/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCdense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAdense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDpolicy/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBpolicy/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvalue/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAvalue/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBconv/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@conv/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCdense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAdense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDpolicy/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBpolicy/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvalue/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAvalue/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÖ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesâ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityŤ
AssignVariableOpAssignVariableOp,assignvariableop_tic_tac_toe_net_conv_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ą
AssignVariableOp_1AssignVariableOp,assignvariableop_1_tic_tac_toe_net_conv_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2´
AssignVariableOp_2AssignVariableOp/assignvariableop_2_tic_tac_toe_net_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3˛
AssignVariableOp_3AssignVariableOp-assignvariableop_3_tic_tac_toe_net_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ľ
AssignVariableOp_4AssignVariableOp0assignvariableop_4_tic_tac_toe_net_policy_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ł
AssignVariableOp_5AssignVariableOp.assignvariableop_5_tic_tac_toe_net_policy_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6´
AssignVariableOp_6AssignVariableOp/assignvariableop_6_tic_tac_toe_net_value_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7˛
AssignVariableOp_7AssignVariableOp-assignvariableop_7_tic_tac_toe_net_value_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8Ą
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ł
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10§
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ś
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ž
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ą
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ą
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ł
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ł
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ł
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ł
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19ž
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_tic_tac_toe_net_conv_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20ź
AssignVariableOp_20AssignVariableOp4assignvariableop_20_adam_tic_tac_toe_net_conv_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ż
AssignVariableOp_21AssignVariableOp7assignvariableop_21_adam_tic_tac_toe_net_dense_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22˝
AssignVariableOp_22AssignVariableOp5assignvariableop_22_adam_tic_tac_toe_net_dense_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ŕ
AssignVariableOp_23AssignVariableOp8assignvariableop_23_adam_tic_tac_toe_net_policy_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24ž
AssignVariableOp_24AssignVariableOp6assignvariableop_24_adam_tic_tac_toe_net_policy_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25ż
AssignVariableOp_25AssignVariableOp7assignvariableop_25_adam_tic_tac_toe_net_value_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26˝
AssignVariableOp_26AssignVariableOp5assignvariableop_26_adam_tic_tac_toe_net_value_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27ž
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_tic_tac_toe_net_conv_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28ź
AssignVariableOp_28AssignVariableOp4assignvariableop_28_adam_tic_tac_toe_net_conv_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29ż
AssignVariableOp_29AssignVariableOp7assignvariableop_29_adam_tic_tac_toe_net_dense_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30˝
AssignVariableOp_30AssignVariableOp5assignvariableop_30_adam_tic_tac_toe_net_dense_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ŕ
AssignVariableOp_31AssignVariableOp8assignvariableop_31_adam_tic_tac_toe_net_policy_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32ž
AssignVariableOp_32AssignVariableOp6assignvariableop_32_adam_tic_tac_toe_net_policy_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33ż
AssignVariableOp_33AssignVariableOp7assignvariableop_33_adam_tic_tac_toe_net_value_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34˝
AssignVariableOp_34AssignVariableOp5assignvariableop_34_adam_tic_tac_toe_net_value_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_349
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpŕ
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_35Ó
Identity_36IdentityIdentity_35:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_36"#
identity_36Identity_36:output:0*Ł
_input_shapes
: :::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342(
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
ś
Ž
F__inference_policy_layer_call_and_return_conditional_losses_1973734604

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:		*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙	2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙	2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙	2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙	2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ŕ

*__inference_value_layer_call_fn_1973734797

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallő
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_value_layer_call_and_return_conditional_losses_19737346312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
˘

ô
4__inference_tic_tac_toe_net_layer_call_fn_1973734673
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1˘StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙	:˙˙˙˙˙˙˙˙˙**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_tic_tac_toe_net_layer_call_and_return_conditional_losses_19737346492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙	2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*N
_input_shapes=
;:˙˙˙˙˙˙˙˙˙::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
ł
­
E__inference_dense_layer_call_and_return_conditional_losses_1973734577

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Á
c
G__inference_flatten_layer_call_and_return_conditional_losses_1973734558

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
Ź
D__inference_conv_layer_call_and_return_conditional_losses_1973734536

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpĽ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙:::W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ń
serving_defaultÝ
C
input_18
serving_default_input_1:0˙˙˙˙˙˙˙˙˙<
output_10
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙	<
output_20
StatefulPartitionedCall:1˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:Ö

conv
flatten
	dense

policy
	value
	optimizer
loss
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
k_default_save_signature
l__call__
*m&call_and_return_all_conditional_losses"ů
_tf_keras_modelß{"class_name": "TicTacToeNet", "name": "tic_tac_toe_net", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "TicTacToeNet"}, "training_config": {"loss": ["categorical_crossentropy", "mean_squared_error"], "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ĺ	

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
n__call__
*o&call_and_return_all_conditional_losses"Ŕ
_tf_keras_layerŚ{"class_name": "Conv2D", "name": "conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 3, 3, 1]}}
â
trainable_variables
regularization_losses
	variables
	keras_api
p__call__
*q&call_and_return_all_conditional_losses"Ó
_tf_keras_layerš{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ě

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
r__call__
*s&call_and_return_all_conditional_losses"Ç
_tf_keras_layer­{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128]}}
ď

kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
t__call__
*u&call_and_return_all_conditional_losses"Ę
_tf_keras_layer°{"class_name": "Dense", "name": "policy", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "policy", "trainable": true, "dtype": "float32", "units": 9, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512]}}
ę

#kernel
$bias
%trainable_variables
&regularization_losses
'	variables
(	keras_api
v__call__
*w&call_and_return_all_conditional_losses"Ĺ
_tf_keras_layerŤ{"class_name": "Dense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512]}}
ă
)iter

*beta_1

+beta_2
	,decay
-learning_ratem[m\m]m^m_m`#ma$mbvcvdvevfvgvh#vi$vj"
	optimizer
 "
trackable_list_wrapper
X
0
1
2
3
4
5
#6
$7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
#6
$7"
trackable_list_wrapper
Ę
.layer_regularization_losses
/layer_metrics
0non_trainable_variables

1layers
trainable_variables
	regularization_losses

	variables
2metrics
l__call__
k_default_save_signature
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
,
xserving_default"
signature_map
6:42tic_tac_toe_net/conv/kernel
(:&2tic_tac_toe_net/conv/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
3layer_metrics
4non_trainable_variables

5layers
trainable_variables
regularization_losses
6layer_regularization_losses
	variables
7metrics
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
8layer_metrics
9non_trainable_variables

:layers
trainable_variables
regularization_losses
;layer_regularization_losses
	variables
<metrics
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
0:.
2tic_tac_toe_net/dense/kernel
):'2tic_tac_toe_net/dense/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
=layer_metrics
>non_trainable_variables

?layers
trainable_variables
regularization_losses
@layer_regularization_losses
	variables
Ametrics
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
0:.		2tic_tac_toe_net/policy/kernel
):'	2tic_tac_toe_net/policy/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Blayer_metrics
Cnon_trainable_variables

Dlayers
trainable_variables
 regularization_losses
Elayer_regularization_losses
!	variables
Fmetrics
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
/:-	2tic_tac_toe_net/value/kernel
(:&2tic_tac_toe_net/value/bias
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
­
Glayer_metrics
Hnon_trainable_variables

Ilayers
%trainable_variables
&regularization_losses
Jlayer_regularization_losses
'	variables
Kmetrics
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
5
L0
M1
N2"
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
ť
	Ototal
	Pcount
Q	variables
R	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
Í
	Stotal
	Tcount
U	variables
V	keras_api"
_tf_keras_metric|{"class_name": "Mean", "name": "output_1_loss", "dtype": "float32", "config": {"name": "output_1_loss", "dtype": "float32"}}
Í
	Wtotal
	Xcount
Y	variables
Z	keras_api"
_tf_keras_metric|{"class_name": "Mean", "name": "output_2_loss", "dtype": "float32", "config": {"name": "output_2_loss", "dtype": "float32"}}
:  (2total
:  (2count
.
O0
P1"
trackable_list_wrapper
-
Q	variables"
_generic_user_object
:  (2total
:  (2count
.
S0
T1"
trackable_list_wrapper
-
U	variables"
_generic_user_object
:  (2total
:  (2count
.
W0
X1"
trackable_list_wrapper
-
Y	variables"
_generic_user_object
;:92"Adam/tic_tac_toe_net/conv/kernel/m
-:+2 Adam/tic_tac_toe_net/conv/bias/m
5:3
2#Adam/tic_tac_toe_net/dense/kernel/m
.:,2!Adam/tic_tac_toe_net/dense/bias/m
5:3		2$Adam/tic_tac_toe_net/policy/kernel/m
.:,	2"Adam/tic_tac_toe_net/policy/bias/m
4:2	2#Adam/tic_tac_toe_net/value/kernel/m
-:+2!Adam/tic_tac_toe_net/value/bias/m
;:92"Adam/tic_tac_toe_net/conv/kernel/v
-:+2 Adam/tic_tac_toe_net/conv/bias/v
5:3
2#Adam/tic_tac_toe_net/dense/kernel/v
.:,2!Adam/tic_tac_toe_net/dense/bias/v
5:3		2$Adam/tic_tac_toe_net/policy/kernel/v
.:,	2"Adam/tic_tac_toe_net/policy/bias/v
4:2	2#Adam/tic_tac_toe_net/value/kernel/v
-:+2!Adam/tic_tac_toe_net/value/bias/v
ë2č
%__inference__wrapped_model_1973734521ž
˛
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *.˘+
)&
input_1˙˙˙˙˙˙˙˙˙
2
4__inference_tic_tac_toe_net_layer_call_fn_1973734673Î
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
annotationsŞ *.˘+
)&
input_1˙˙˙˙˙˙˙˙˙
Ľ2˘
O__inference_tic_tac_toe_net_layer_call_and_return_conditional_losses_1973734649Î
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
annotationsŞ *.˘+
)&
input_1˙˙˙˙˙˙˙˙˙
Ó2Đ
)__inference_conv_layer_call_fn_1973734726˘
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
î2ë
D__inference_conv_layer_call_and_return_conditional_losses_1973734717˘
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
Ö2Ó
,__inference_flatten_layer_call_fn_1973734737˘
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
ń2î
G__inference_flatten_layer_call_and_return_conditional_losses_1973734732˘
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
Ô2Ń
*__inference_dense_layer_call_fn_1973734757˘
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
ď2ě
E__inference_dense_layer_call_and_return_conditional_losses_1973734748˘
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
Ő2Ň
+__inference_policy_layer_call_fn_1973734777˘
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
đ2í
F__inference_policy_layer_call_and_return_conditional_losses_1973734768˘
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
Ô2Ń
*__inference_value_layer_call_fn_1973734797˘
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
ď2ě
E__inference_value_layer_call_and_return_conditional_losses_1973734788˘
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
7B5
(__inference_signature_wrapper_1973734706input_1Ó
%__inference__wrapped_model_1973734521Š#$8˘5
.˘+
)&
input_1˙˙˙˙˙˙˙˙˙
Ş "cŞ`
.
output_1"
output_1˙˙˙˙˙˙˙˙˙	
.
output_2"
output_2˙˙˙˙˙˙˙˙˙ľ
D__inference_conv_layer_call_and_return_conditional_losses_1973734717m7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙
 
)__inference_conv_layer_call_fn_1973734726`7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş "!˙˙˙˙˙˙˙˙˙§
E__inference_dense_layer_call_and_return_conditional_losses_1973734748^0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
*__inference_dense_layer_call_fn_1973734757Q0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙­
G__inference_flatten_layer_call_and_return_conditional_losses_1973734732b8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
,__inference_flatten_layer_call_fn_1973734737U8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙§
F__inference_policy_layer_call_and_return_conditional_losses_1973734768]0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙	
 
+__inference_policy_layer_call_fn_1973734777P0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙	á
(__inference_signature_wrapper_1973734706´#$C˘@
˘ 
9Ş6
4
input_1)&
input_1˙˙˙˙˙˙˙˙˙"cŞ`
.
output_1"
output_1˙˙˙˙˙˙˙˙˙	
.
output_2"
output_2˙˙˙˙˙˙˙˙˙ĺ
O__inference_tic_tac_toe_net_layer_call_and_return_conditional_losses_1973734649#$8˘5
.˘+
)&
input_1˙˙˙˙˙˙˙˙˙
Ş "K˘H
A˘>

0/0˙˙˙˙˙˙˙˙˙	

0/1˙˙˙˙˙˙˙˙˙
 ź
4__inference_tic_tac_toe_net_layer_call_fn_1973734673#$8˘5
.˘+
)&
input_1˙˙˙˙˙˙˙˙˙
Ş "=˘:

0˙˙˙˙˙˙˙˙˙	

1˙˙˙˙˙˙˙˙˙Ś
E__inference_value_layer_call_and_return_conditional_losses_1973734788]#$0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ~
*__inference_value_layer_call_fn_1973734797P#$0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙