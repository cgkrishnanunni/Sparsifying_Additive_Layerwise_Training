??
??
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
dtypetype?
?
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
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.0.02unknown8??
z
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shared_namedense_20/kernel*
shape
:d
s
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
dtype0*
_output_shapes

:d
r
dense_20/biasVarHandleOp*
shared_namedense_20/bias*
shape:d*
_output_shapes
: *
dtype0
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes
:d*
dtype0
z
dense_21/kernelVarHandleOp*
shape
:dd*
_output_shapes
: * 
shared_namedense_21/kernel*
dtype0
s
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
dtype0*
_output_shapes

:dd
r
dense_21/biasVarHandleOp*
shared_namedense_21/bias*
dtype0*
_output_shapes
: *
shape:d
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes
:d*
dtype0
z
dense_22/kernelVarHandleOp*
dtype0*
shape
:dd* 
shared_namedense_22/kernel*
_output_shapes
: 
s
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes

:dd*
dtype0
r
dense_22/biasVarHandleOp*
dtype0*
shape:d*
shared_namedense_22/bias*
_output_shapes
: 
k
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes
:d*
dtype0
z
dense_23/kernelVarHandleOp* 
shared_namedense_23/kernel*
dtype0*
_output_shapes
: *
shape
:dd
s
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
dtype0*
_output_shapes

:dd
r
dense_23/biasVarHandleOp*
_output_shapes
: *
shape:d*
shared_namedense_23/bias*
dtype0
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
dtype0*
_output_shapes
:d
z
dense_24/kernelVarHandleOp*
shape
:dd*
_output_shapes
: *
dtype0* 
shared_namedense_24/kernel
s
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes

:dd*
dtype0
r
dense_24/biasVarHandleOp*
_output_shapes
: *
shared_namedense_24/bias*
dtype0*
shape:d
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
dtype0*
_output_shapes
:d
z
dense_25/kernelVarHandleOp*
shape
:dd* 
shared_namedense_25/kernel*
_output_shapes
: *
dtype0
s
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel*
_output_shapes

:dd*
dtype0
r
dense_25/biasVarHandleOp*
_output_shapes
: *
shape:d*
shared_namedense_25/bias*
dtype0
k
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
dtype0*
_output_shapes
:d
z
dense_26/kernelVarHandleOp* 
shared_namedense_26/kernel*
_output_shapes
: *
dtype0*
shape
:dd
s
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
dtype0*
_output_shapes

:dd
r
dense_26/biasVarHandleOp*
dtype0*
shared_namedense_26/bias*
shape:d*
_output_shapes
: 
k
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
_output_shapes
:d*
dtype0
z
dense_27/kernelVarHandleOp*
shape
:dd*
_output_shapes
: * 
shared_namedense_27/kernel*
dtype0
s
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel*
_output_shapes

:dd*
dtype0
r
dense_27/biasVarHandleOp*
shared_namedense_27/bias*
shape:d*
dtype0*
_output_shapes
: 
k
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes
:d*
dtype0
z
dense_28/kernelVarHandleOp*
_output_shapes
: * 
shared_namedense_28/kernel*
dtype0*
shape
:dd
s
#dense_28/kernel/Read/ReadVariableOpReadVariableOpdense_28/kernel*
dtype0*
_output_shapes

:dd
r
dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shared_namedense_28/bias*
shape:d
k
!dense_28/bias/Read/ReadVariableOpReadVariableOpdense_28/bias*
dtype0*
_output_shapes
:d
z
dense_29/kernelVarHandleOp*
dtype0*
_output_shapes
: * 
shared_namedense_29/kernel*
shape
:d
s
#dense_29/kernel/Read/ReadVariableOpReadVariableOpdense_29/kernel*
dtype0*
_output_shapes

:d
r
dense_29/biasVarHandleOp*
shape:*
dtype0*
shared_namedense_29/bias*
_output_shapes
: 
k
!dense_29/bias/Read/ReadVariableOpReadVariableOpdense_29/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
shape: *
_output_shapes
: *
shared_name	Adam/iter*
dtype0	
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
dtype0	*
_output_shapes
: 
j
Adam/beta_1VarHandleOp*
dtype0*
shape: *
_output_shapes
: *
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
shape: *
shared_nameAdam/beta_2*
dtype0
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
shape: *
_output_shapes
: *
dtype0*
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
shape: *#
shared_nameAdam/learning_rate*
dtype0*
_output_shapes
: 
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
dtype0*
shared_nametotal*
shape: *
_output_shapes
: 
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
shape: *
dtype0*
shared_namecount*
_output_shapes
: 
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
?
Adam/dense_20/kernel/mVarHandleOp*
shape
:d*
_output_shapes
: *'
shared_nameAdam/dense_20/kernel/m*
dtype0
?
*Adam/dense_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/m*
dtype0*
_output_shapes

:d
?
Adam/dense_20/bias/mVarHandleOp*
shape:d*%
shared_nameAdam/dense_20/bias/m*
_output_shapes
: *
dtype0
y
(Adam/dense_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/m*
dtype0*
_output_shapes
:d
?
Adam/dense_21/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape
:dd*'
shared_nameAdam/dense_21/kernel/m
?
*Adam/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/m*
dtype0*
_output_shapes

:dd
?
Adam/dense_21/bias/mVarHandleOp*
dtype0*%
shared_nameAdam/dense_21/bias/m*
shape:d*
_output_shapes
: 
y
(Adam/dense_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/m*
dtype0*
_output_shapes
:d
?
Adam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*'
shared_nameAdam/dense_22/kernel/m*
shape
:dd
?
*Adam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/m*
dtype0*
_output_shapes

:dd
?
Adam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
shape:d*
dtype0*%
shared_nameAdam/dense_22/bias/m
y
(Adam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_23/kernel/mVarHandleOp*
dtype0*
shape
:dd*'
shared_nameAdam/dense_23/kernel/m*
_output_shapes
: 
?
*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m*
_output_shapes

:dd*
dtype0
?
Adam/dense_23/bias/mVarHandleOp*
shape:d*
_output_shapes
: *
dtype0*%
shared_nameAdam/dense_23/bias/m
y
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_24/kernel/mVarHandleOp*
shape
:dd*'
shared_nameAdam/dense_24/kernel/m*
_output_shapes
: *
dtype0
?
*Adam/dense_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/m*
_output_shapes

:dd*
dtype0
?
Adam/dense_24/bias/mVarHandleOp*
_output_shapes
: *
shape:d*
dtype0*%
shared_nameAdam/dense_24/bias/m
y
(Adam/dense_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_25/kernel/mVarHandleOp*
_output_shapes
: *
shape
:dd*
dtype0*'
shared_nameAdam/dense_25/kernel/m
?
*Adam/dense_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/m*
_output_shapes

:dd*
dtype0
?
Adam/dense_25/bias/mVarHandleOp*
_output_shapes
: *%
shared_nameAdam/dense_25/bias/m*
shape:d*
dtype0
y
(Adam/dense_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_26/kernel/mVarHandleOp*
shape
:dd*
dtype0*'
shared_nameAdam/dense_26/kernel/m*
_output_shapes
: 
?
*Adam/dense_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/m*
_output_shapes

:dd*
dtype0
?
Adam/dense_26/bias/mVarHandleOp*
_output_shapes
: *
shape:d*%
shared_nameAdam/dense_26/bias/m*
dtype0
y
(Adam/dense_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_27/kernel/mVarHandleOp*'
shared_nameAdam/dense_27/kernel/m*
shape
:dd*
_output_shapes
: *
dtype0
?
*Adam/dense_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/m*
_output_shapes

:dd*
dtype0
?
Adam/dense_27/bias/mVarHandleOp*%
shared_nameAdam/dense_27/bias/m*
dtype0*
shape:d*
_output_shapes
: 
y
(Adam/dense_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_28/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *'
shared_nameAdam/dense_28/kernel/m*
shape
:dd
?
*Adam/dense_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/m*
_output_shapes

:dd*
dtype0
?
Adam/dense_28/bias/mVarHandleOp*
dtype0*
_output_shapes
: *%
shared_nameAdam/dense_28/bias/m*
shape:d
y
(Adam/dense_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/m*
dtype0*
_output_shapes
:d
?
Adam/dense_29/kernel/mVarHandleOp*'
shared_nameAdam/dense_29/kernel/m*
_output_shapes
: *
shape
:d*
dtype0
?
*Adam/dense_29/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/m*
_output_shapes

:d*
dtype0
?
Adam/dense_29/bias/mVarHandleOp*%
shared_nameAdam/dense_29/bias/m*
dtype0*
_output_shapes
: *
shape:
y
(Adam/dense_29/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/m*
dtype0*
_output_shapes
:
?
Adam/dense_20/kernel/vVarHandleOp*
dtype0*
shape
:d*'
shared_nameAdam/dense_20/kernel/v*
_output_shapes
: 
?
*Adam/dense_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/v*
_output_shapes

:d*
dtype0
?
Adam/dense_20/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:d*%
shared_nameAdam/dense_20/bias/v
y
(Adam/dense_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_21/kernel/vVarHandleOp*'
shared_nameAdam/dense_21/kernel/v*
dtype0*
shape
:dd*
_output_shapes
: 
?
*Adam/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/v*
_output_shapes

:dd*
dtype0
?
Adam/dense_21/bias/vVarHandleOp*
dtype0*%
shared_nameAdam/dense_21/bias/v*
_output_shapes
: *
shape:d
y
(Adam/dense_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_22/kernel/vVarHandleOp*
dtype0*'
shared_nameAdam/dense_22/kernel/v*
shape
:dd*
_output_shapes
: 
?
*Adam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/v*
_output_shapes

:dd*
dtype0
?
Adam/dense_22/bias/vVarHandleOp*
shape:d*
_output_shapes
: *
dtype0*%
shared_nameAdam/dense_22/bias/v
y
(Adam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_23/kernel/vVarHandleOp*
dtype0*'
shared_nameAdam/dense_23/kernel/v*
_output_shapes
: *
shape
:dd
?
*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v*
_output_shapes

:dd*
dtype0
?
Adam/dense_23/bias/vVarHandleOp*
dtype0*%
shared_nameAdam/dense_23/bias/v*
shape:d*
_output_shapes
: 
y
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_24/kernel/vVarHandleOp*'
shared_nameAdam/dense_24/kernel/v*
shape
:dd*
_output_shapes
: *
dtype0
?
*Adam/dense_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/v*
_output_shapes

:dd*
dtype0
?
Adam/dense_24/bias/vVarHandleOp*%
shared_nameAdam/dense_24/bias/v*
shape:d*
_output_shapes
: *
dtype0
y
(Adam/dense_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_25/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape
:dd*'
shared_nameAdam/dense_25/kernel/v
?
*Adam/dense_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/v*
dtype0*
_output_shapes

:dd
?
Adam/dense_25/bias/vVarHandleOp*%
shared_nameAdam/dense_25/bias/v*
shape:d*
dtype0*
_output_shapes
: 
y
(Adam/dense_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/v*
dtype0*
_output_shapes
:d
?
Adam/dense_26/kernel/vVarHandleOp*'
shared_nameAdam/dense_26/kernel/v*
_output_shapes
: *
dtype0*
shape
:dd
?
*Adam/dense_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/v*
dtype0*
_output_shapes

:dd
?
Adam/dense_26/bias/vVarHandleOp*%
shared_nameAdam/dense_26/bias/v*
_output_shapes
: *
shape:d*
dtype0
y
(Adam/dense_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_27/kernel/vVarHandleOp*
shape
:dd*
dtype0*
_output_shapes
: *'
shared_nameAdam/dense_27/kernel/v
?
*Adam/dense_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/v*
_output_shapes

:dd*
dtype0
?
Adam/dense_27/bias/vVarHandleOp*%
shared_nameAdam/dense_27/bias/v*
_output_shapes
: *
dtype0*
shape:d
y
(Adam/dense_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/v*
dtype0*
_output_shapes
:d
?
Adam/dense_28/kernel/vVarHandleOp*
_output_shapes
: *
shape
:dd*'
shared_nameAdam/dense_28/kernel/v*
dtype0
?
*Adam/dense_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/v*
dtype0*
_output_shapes

:dd
?
Adam/dense_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*%
shared_nameAdam/dense_28/bias/v*
shape:d
y
(Adam/dense_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/v*
dtype0*
_output_shapes
:d
?
Adam/dense_29/kernel/vVarHandleOp*
dtype0*
shape
:d*'
shared_nameAdam/dense_29/kernel/v*
_output_shapes
: 
?
*Adam/dense_29/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/v*
dtype0*
_output_shapes

:d
?
Adam/dense_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_29/bias/v
y
(Adam/dense_29/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/v*
dtype0*
_output_shapes
:

NoOpNoOp
?c
ConstConst"/device:CPU:0*
dtype0*?b
value?bB?b B?b
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
 	variables
!	keras_api
h

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
h

.kernel
/bias
0regularization_losses
1trainable_variables
2	variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api
h

:kernel
;bias
<regularization_losses
=trainable_variables
>	variables
?	keras_api
h

@kernel
Abias
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
h

Fkernel
Gbias
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
h

Lkernel
Mbias
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
?
Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratem?m?m?m?"m?#m?(m?)m?.m?/m?4m?5m?:m?;m?@m?Am?Fm?Gm?Lm?Mm?v?v?v?v?"v?#v?(v?)v?.v?/v?4v?5v?:v?;v?@v?Av?Fv?Gv?Lv?Mv?
 
?
0
1
2
3
"4
#5
(6
)7
.8
/9
410
511
:12
;13
@14
A15
F16
G17
L18
M19
?
0
1
2
3
"4
#5
(6
)7
.8
/9
410
511
:12
;13
@14
A15
F16
G17
L18
M19
?
Wmetrics
regularization_losses
Xnon_trainable_variables
trainable_variables
Ylayer_regularization_losses

Zlayers
	variables
 
 
 
 
?
[metrics
regularization_losses
\non_trainable_variables
trainable_variables
]layer_regularization_losses

^layers
	variables
[Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_20/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
_metrics
regularization_losses
`non_trainable_variables
trainable_variables
alayer_regularization_losses

blayers
	variables
[Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_21/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
cmetrics
regularization_losses
dnon_trainable_variables
trainable_variables
elayer_regularization_losses

flayers
 	variables
[Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_22/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
?
gmetrics
$regularization_losses
hnon_trainable_variables
%trainable_variables
ilayer_regularization_losses

jlayers
&	variables
[Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_23/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
?
kmetrics
*regularization_losses
lnon_trainable_variables
+trainable_variables
mlayer_regularization_losses

nlayers
,	variables
[Y
VARIABLE_VALUEdense_24/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_24/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
?
ometrics
0regularization_losses
pnon_trainable_variables
1trainable_variables
qlayer_regularization_losses

rlayers
2	variables
[Y
VARIABLE_VALUEdense_25/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_25/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
?
smetrics
6regularization_losses
tnon_trainable_variables
7trainable_variables
ulayer_regularization_losses

vlayers
8	variables
[Y
VARIABLE_VALUEdense_26/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_26/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

:0
;1

:0
;1
?
wmetrics
<regularization_losses
xnon_trainable_variables
=trainable_variables
ylayer_regularization_losses

zlayers
>	variables
[Y
VARIABLE_VALUEdense_27/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_27/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

@0
A1

@0
A1
?
{metrics
Bregularization_losses
|non_trainable_variables
Ctrainable_variables
}layer_regularization_losses

~layers
D	variables
[Y
VARIABLE_VALUEdense_28/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_28/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

F0
G1

F0
G1
?
metrics
Hregularization_losses
?non_trainable_variables
Itrainable_variables
 ?layer_regularization_losses
?layers
J	variables
[Y
VARIABLE_VALUEdense_29/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_29/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1

L0
M1
?
?metrics
Nregularization_losses
?non_trainable_variables
Otrainable_variables
 ?layer_regularization_losses
?layers
P	variables
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

?0
 
 
F
0
1
2
3
4
5
6
	7

8
9
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


?total

?count
?
_fn_kwargs
?regularization_losses
?trainable_variables
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

?0
?1
?
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
 

?0
?1
 
 
~|
VARIABLE_VALUEAdam/dense_20/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_20/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_21/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_21/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_22/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_22/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_23/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_23/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_24/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_24/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_25/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_25/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_26/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_26/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_27/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_27/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_28/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_28/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_29/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_29/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_20/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_20/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_21/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_21/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_22/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_22/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_23/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_23/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_24/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_24/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_25/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_25/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_26/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_26/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_27/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_27/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_28/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_28/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_29/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_29/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
: 
?
serving_default_dense_20_inputPlaceholder*
dtype0*'
_output_shapes
:?????????*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_20_inputdense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/biasdense_26/kerneldense_26/biasdense_27/kerneldense_27/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/bias*,
_gradient_op_typePartitionedCall-13002*'
_output_shapes
:?????????*
Tout
2*,
f'R%
#__inference_signature_wrapper_12490* 
Tin
2**
config_proto

GPU 

CPU2J 8
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOp#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOp#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOp#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOp#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOp#dense_28/kernel/Read/ReadVariableOp!dense_28/bias/Read/ReadVariableOp#dense_29/kernel/Read/ReadVariableOp!dense_29/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_20/kernel/m/Read/ReadVariableOp(Adam/dense_20/bias/m/Read/ReadVariableOp*Adam/dense_21/kernel/m/Read/ReadVariableOp(Adam/dense_21/bias/m/Read/ReadVariableOp*Adam/dense_22/kernel/m/Read/ReadVariableOp(Adam/dense_22/bias/m/Read/ReadVariableOp*Adam/dense_23/kernel/m/Read/ReadVariableOp(Adam/dense_23/bias/m/Read/ReadVariableOp*Adam/dense_24/kernel/m/Read/ReadVariableOp(Adam/dense_24/bias/m/Read/ReadVariableOp*Adam/dense_25/kernel/m/Read/ReadVariableOp(Adam/dense_25/bias/m/Read/ReadVariableOp*Adam/dense_26/kernel/m/Read/ReadVariableOp(Adam/dense_26/bias/m/Read/ReadVariableOp*Adam/dense_27/kernel/m/Read/ReadVariableOp(Adam/dense_27/bias/m/Read/ReadVariableOp*Adam/dense_28/kernel/m/Read/ReadVariableOp(Adam/dense_28/bias/m/Read/ReadVariableOp*Adam/dense_29/kernel/m/Read/ReadVariableOp(Adam/dense_29/bias/m/Read/ReadVariableOp*Adam/dense_20/kernel/v/Read/ReadVariableOp(Adam/dense_20/bias/v/Read/ReadVariableOp*Adam/dense_21/kernel/v/Read/ReadVariableOp(Adam/dense_21/bias/v/Read/ReadVariableOp*Adam/dense_22/kernel/v/Read/ReadVariableOp(Adam/dense_22/bias/v/Read/ReadVariableOp*Adam/dense_23/kernel/v/Read/ReadVariableOp(Adam/dense_23/bias/v/Read/ReadVariableOp*Adam/dense_24/kernel/v/Read/ReadVariableOp(Adam/dense_24/bias/v/Read/ReadVariableOp*Adam/dense_25/kernel/v/Read/ReadVariableOp(Adam/dense_25/bias/v/Read/ReadVariableOp*Adam/dense_26/kernel/v/Read/ReadVariableOp(Adam/dense_26/bias/v/Read/ReadVariableOp*Adam/dense_27/kernel/v/Read/ReadVariableOp(Adam/dense_27/bias/v/Read/ReadVariableOp*Adam/dense_28/kernel/v/Read/ReadVariableOp(Adam/dense_28/bias/v/Read/ReadVariableOp*Adam/dense_29/kernel/v/Read/ReadVariableOp(Adam/dense_29/bias/v/Read/ReadVariableOpConst*,
_gradient_op_typePartitionedCall-13091**
config_proto

GPU 

CPU2J 8*
_output_shapes
: *
Tout
2*P
TinI
G2E	*'
f"R 
__inference__traced_save_13090
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/biasdense_26/kerneldense_26/biasdense_27/kerneldense_27/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_20/kernel/mAdam/dense_20/bias/mAdam/dense_21/kernel/mAdam/dense_21/bias/mAdam/dense_22/kernel/mAdam/dense_22/bias/mAdam/dense_23/kernel/mAdam/dense_23/bias/mAdam/dense_24/kernel/mAdam/dense_24/bias/mAdam/dense_25/kernel/mAdam/dense_25/bias/mAdam/dense_26/kernel/mAdam/dense_26/bias/mAdam/dense_27/kernel/mAdam/dense_27/bias/mAdam/dense_28/kernel/mAdam/dense_28/bias/mAdam/dense_29/kernel/mAdam/dense_29/bias/mAdam/dense_20/kernel/vAdam/dense_20/bias/vAdam/dense_21/kernel/vAdam/dense_21/bias/vAdam/dense_22/kernel/vAdam/dense_22/bias/vAdam/dense_23/kernel/vAdam/dense_23/bias/vAdam/dense_24/kernel/vAdam/dense_24/bias/vAdam/dense_25/kernel/vAdam/dense_25/bias/vAdam/dense_26/kernel/vAdam/dense_26/bias/vAdam/dense_27/kernel/vAdam/dense_27/bias/vAdam/dense_28/kernel/vAdam/dense_28/bias/vAdam/dense_29/kernel/vAdam/dense_29/bias/v*
_output_shapes
: **
f%R#
!__inference__traced_restore_13304**
config_proto

GPU 

CPU2J 8*
Tout
2*,
_gradient_op_typePartitionedCall-13305*O
TinH
F2D??	
?7
?

G__inference_sequential_2_layer_call_and_return_conditional_losses_12300
dense_20_input+
'dense_20_statefulpartitionedcall_args_1+
'dense_20_statefulpartitionedcall_args_2+
'dense_21_statefulpartitionedcall_args_1+
'dense_21_statefulpartitionedcall_args_2+
'dense_22_statefulpartitionedcall_args_1+
'dense_22_statefulpartitionedcall_args_2+
'dense_23_statefulpartitionedcall_args_1+
'dense_23_statefulpartitionedcall_args_2+
'dense_24_statefulpartitionedcall_args_1+
'dense_24_statefulpartitionedcall_args_2+
'dense_25_statefulpartitionedcall_args_1+
'dense_25_statefulpartitionedcall_args_2+
'dense_26_statefulpartitionedcall_args_1+
'dense_26_statefulpartitionedcall_args_2+
'dense_27_statefulpartitionedcall_args_1+
'dense_27_statefulpartitionedcall_args_2+
'dense_28_statefulpartitionedcall_args_1+
'dense_28_statefulpartitionedcall_args_2+
'dense_29_statefulpartitionedcall_args_1+
'dense_29_statefulpartitionedcall_args_2
identity?? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall? dense_26/StatefulPartitionedCall? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall?
 dense_20/StatefulPartitionedCallStatefulPartitionedCalldense_20_input'dense_20_statefulpartitionedcall_args_1'dense_20_statefulpartitionedcall_args_2*
Tin
2**
config_proto

GPU 

CPU2J 8*
Tout
2*'
_output_shapes
:?????????d*,
_gradient_op_typePartitionedCall-12037*L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_12031?
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0'dense_21_statefulpartitionedcall_args_1'dense_21_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12065*
Tin
2**
config_proto

GPU 

CPU2J 8*
Tout
2*L
fGRE
C__inference_dense_21_layer_call_and_return_conditional_losses_12059*'
_output_shapes
:?????????d?
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0'dense_22_statefulpartitionedcall_args_1'dense_22_statefulpartitionedcall_args_2*L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_12087*
Tin
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-12093*'
_output_shapes
:?????????d*
Tout
2?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0'dense_23_statefulpartitionedcall_args_1'dense_23_statefulpartitionedcall_args_2*
Tout
2*
Tin
2*L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_12115*,
_gradient_op_typePartitionedCall-12121**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????d?
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0'dense_24_statefulpartitionedcall_args_1'dense_24_statefulpartitionedcall_args_2*L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_12143*
Tout
2*,
_gradient_op_typePartitionedCall-12149*
Tin
2*'
_output_shapes
:?????????d**
config_proto

GPU 

CPU2J 8?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0'dense_25_statefulpartitionedcall_args_1'dense_25_statefulpartitionedcall_args_2*L
fGRE
C__inference_dense_25_layer_call_and_return_conditional_losses_12171*'
_output_shapes
:?????????d**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-12177*
Tin
2*
Tout
2?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0'dense_26_statefulpartitionedcall_args_1'dense_26_statefulpartitionedcall_args_2*'
_output_shapes
:?????????d*
Tin
2*
Tout
2*L
fGRE
C__inference_dense_26_layer_call_and_return_conditional_losses_12199*,
_gradient_op_typePartitionedCall-12205**
config_proto

GPU 

CPU2J 8?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0'dense_27_statefulpartitionedcall_args_1'dense_27_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12233*
Tin
2*
Tout
2*L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_12227*'
_output_shapes
:?????????d**
config_proto

GPU 

CPU2J 8?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0'dense_28_statefulpartitionedcall_args_1'dense_28_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12261**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_28_layer_call_and_return_conditional_losses_12255*
Tin
2*'
_output_shapes
:?????????d*
Tout
2?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0'dense_29_statefulpartitionedcall_args_1'dense_29_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12288*
Tout
2*L
fGRE
C__inference_dense_29_layer_call_and_return_conditional_losses_12282*
Tin
2*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8?
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*v
_input_shapese
c:?????????::::::::::::::::::::2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall: : : : : : : : : : :. *
(
_user_specified_namedense_20_input: : : : : : : : :	 :
 
?7
?

G__inference_sequential_2_layer_call_and_return_conditional_losses_12336
dense_20_input+
'dense_20_statefulpartitionedcall_args_1+
'dense_20_statefulpartitionedcall_args_2+
'dense_21_statefulpartitionedcall_args_1+
'dense_21_statefulpartitionedcall_args_2+
'dense_22_statefulpartitionedcall_args_1+
'dense_22_statefulpartitionedcall_args_2+
'dense_23_statefulpartitionedcall_args_1+
'dense_23_statefulpartitionedcall_args_2+
'dense_24_statefulpartitionedcall_args_1+
'dense_24_statefulpartitionedcall_args_2+
'dense_25_statefulpartitionedcall_args_1+
'dense_25_statefulpartitionedcall_args_2+
'dense_26_statefulpartitionedcall_args_1+
'dense_26_statefulpartitionedcall_args_2+
'dense_27_statefulpartitionedcall_args_1+
'dense_27_statefulpartitionedcall_args_2+
'dense_28_statefulpartitionedcall_args_1+
'dense_28_statefulpartitionedcall_args_2+
'dense_29_statefulpartitionedcall_args_1+
'dense_29_statefulpartitionedcall_args_2
identity?? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall? dense_26/StatefulPartitionedCall? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall?
 dense_20/StatefulPartitionedCallStatefulPartitionedCalldense_20_input'dense_20_statefulpartitionedcall_args_1'dense_20_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12037**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????d*L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_12031*
Tin
2*
Tout
2?
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0'dense_21_statefulpartitionedcall_args_1'dense_21_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*
Tout
2*L
fGRE
C__inference_dense_21_layer_call_and_return_conditional_losses_12059*'
_output_shapes
:?????????d*,
_gradient_op_typePartitionedCall-12065?
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0'dense_22_statefulpartitionedcall_args_1'dense_22_statefulpartitionedcall_args_2*
Tout
2*'
_output_shapes
:?????????d*,
_gradient_op_typePartitionedCall-12093*
Tin
2**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_12087?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0'dense_23_statefulpartitionedcall_args_1'dense_23_statefulpartitionedcall_args_2*'
_output_shapes
:?????????d*
Tin
2*
Tout
2**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_12115*,
_gradient_op_typePartitionedCall-12121?
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0'dense_24_statefulpartitionedcall_args_1'dense_24_statefulpartitionedcall_args_2*L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_12143**
config_proto

GPU 

CPU2J 8*
Tout
2*
Tin
2*,
_gradient_op_typePartitionedCall-12149*'
_output_shapes
:?????????d?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0'dense_25_statefulpartitionedcall_args_1'dense_25_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12177*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????d*L
fGRE
C__inference_dense_25_layer_call_and_return_conditional_losses_12171?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0'dense_26_statefulpartitionedcall_args_1'dense_26_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????d*L
fGRE
C__inference_dense_26_layer_call_and_return_conditional_losses_12199*
Tout
2*,
_gradient_op_typePartitionedCall-12205*
Tin
2?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0'dense_27_statefulpartitionedcall_args_1'dense_27_statefulpartitionedcall_args_2*'
_output_shapes
:?????????d*
Tout
2*L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_12227*
Tin
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-12233?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0'dense_28_statefulpartitionedcall_args_1'dense_28_statefulpartitionedcall_args_2*
Tout
2*L
fGRE
C__inference_dense_28_layer_call_and_return_conditional_losses_12255*,
_gradient_op_typePartitionedCall-12261*
Tin
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????d?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0'dense_29_statefulpartitionedcall_args_1'dense_29_statefulpartitionedcall_args_2*L
fGRE
C__inference_dense_29_layer_call_and_return_conditional_losses_12282*
Tin
2*
Tout
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-12288*'
_output_shapes
:??????????
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*v
_input_shapese
c:?????????::::::::::::::::::::2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall: : : : : : : :. *
(
_user_specified_namedense_20_input: : : : : : : : :	 :
 : : : 
?	
?
C__inference_dense_22_layer_call_and_return_conditional_losses_12732

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dv
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dN
EluEluBiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : :& "
 
_user_specified_nameinputs
?	
?
C__inference_dense_25_layer_call_and_return_conditional_losses_12171

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0N
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?	
?
C__inference_dense_27_layer_call_and_return_conditional_losses_12822

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:ddi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dv
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dN
EluEluBiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*.
_input_shapes
:?????????d::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?	
?
C__inference_dense_23_layer_call_and_return_conditional_losses_12115

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:ddi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dv
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0N
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????d*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????d::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
(__inference_dense_29_layer_call_fn_12864

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*'
_output_shapes
:?????????*
Tout
2*,
_gradient_op_typePartitionedCall-12288**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_29_layer_call_and_return_conditional_losses_12282*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
C__inference_dense_29_layer_call_and_return_conditional_losses_12282

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:di
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
(__inference_dense_28_layer_call_fn_12847

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????d*
Tout
2*,
_gradient_op_typePartitionedCall-12261*
Tin
2*L
fGRE
C__inference_dense_28_layer_call_and_return_conditional_losses_12255?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
,__inference_sequential_2_layer_call_fn_12686

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-12436*
Tout
2*P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_12435**
config_proto

GPU 

CPU2J 8* 
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*v
_input_shapese
c:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : 
?Y
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_12636

inputs+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource+
'dense_21_matmul_readvariableop_resource,
(dense_21_biasadd_readvariableop_resource+
'dense_22_matmul_readvariableop_resource,
(dense_22_biasadd_readvariableop_resource+
'dense_23_matmul_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource+
'dense_24_matmul_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource+
'dense_25_matmul_readvariableop_resource,
(dense_25_biasadd_readvariableop_resource+
'dense_26_matmul_readvariableop_resource,
(dense_26_biasadd_readvariableop_resource+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource+
'dense_28_matmul_readvariableop_resource,
(dense_28_biasadd_readvariableop_resource+
'dense_29_matmul_readvariableop_resource,
(dense_29_biasadd_readvariableop_resource
identity??dense_20/BiasAdd/ReadVariableOp?dense_20/MatMul/ReadVariableOp?dense_21/BiasAdd/ReadVariableOp?dense_21/MatMul/ReadVariableOp?dense_22/BiasAdd/ReadVariableOp?dense_22/MatMul/ReadVariableOp?dense_23/BiasAdd/ReadVariableOp?dense_23/MatMul/ReadVariableOp?dense_24/BiasAdd/ReadVariableOp?dense_24/MatMul/ReadVariableOp?dense_25/BiasAdd/ReadVariableOp?dense_25/MatMul/ReadVariableOp?dense_26/BiasAdd/ReadVariableOp?dense_26/MatMul/ReadVariableOp?dense_27/BiasAdd/ReadVariableOp?dense_27/MatMul/ReadVariableOp?dense_28/BiasAdd/ReadVariableOp?dense_28/MatMul/ReadVariableOp?dense_29/BiasAdd/ReadVariableOp?dense_29/MatMul/ReadVariableOp?
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:d{
dense_20/MatMulMatMulinputs&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0?
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dd?
dense_21/MatMulMatMuldense_20/BiasAdd:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0?
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d`
dense_21/EluEludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:dd*
dtype0?
dense_22/MatMulMatMuldense_21/Elu:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:d?
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0`
dense_22/EluEludense_22/BiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dd?
dense_23/MatMulMatMuldense_22/Elu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0?
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d`
dense_23/EluEludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:dd*
dtype0?
dense_24/MatMulMatMuldense_23/Elu:activations:0&dense_24/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0?
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d`
dense_24/EluEludense_24/BiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dd?
dense_25/MatMulMatMuldense_24/Elu:activations:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:d?
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0`
dense_25/EluEludense_25/BiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dd?
dense_26/MatMulMatMuldense_25/Elu:activations:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:d?
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0`
dense_26/EluEludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dd?
dense_27/MatMulMatMuldense_26/Elu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:d?
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0`
dense_27/EluEludense_27/BiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:dd*
dtype0?
dense_28/MatMulMatMuldense_27/Elu:activations:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0?
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d`
dense_28/EluEludense_28/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:d?
dense_29/MatMulMatMuldense_28/Elu:activations:0&dense_29/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
IdentityIdentitydense_29/BiasAdd:output:0 ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*v
_input_shapese
c:?????????::::::::::::::::::::2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp: : : :	 :
 : : : : : : : : : : :& "
 
_user_specified_nameinputs: : : : : 
?
?
(__inference_dense_25_layer_call_fn_12793

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*,
_gradient_op_typePartitionedCall-12177*
Tout
2**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_25_layer_call_and_return_conditional_losses_12171*'
_output_shapes
:?????????d?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????d*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?	
?
C__inference_dense_22_layer_call_and_return_conditional_losses_12087

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0N
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?7
?

G__inference_sequential_2_layer_call_and_return_conditional_losses_12435

inputs+
'dense_20_statefulpartitionedcall_args_1+
'dense_20_statefulpartitionedcall_args_2+
'dense_21_statefulpartitionedcall_args_1+
'dense_21_statefulpartitionedcall_args_2+
'dense_22_statefulpartitionedcall_args_1+
'dense_22_statefulpartitionedcall_args_2+
'dense_23_statefulpartitionedcall_args_1+
'dense_23_statefulpartitionedcall_args_2+
'dense_24_statefulpartitionedcall_args_1+
'dense_24_statefulpartitionedcall_args_2+
'dense_25_statefulpartitionedcall_args_1+
'dense_25_statefulpartitionedcall_args_2+
'dense_26_statefulpartitionedcall_args_1+
'dense_26_statefulpartitionedcall_args_2+
'dense_27_statefulpartitionedcall_args_1+
'dense_27_statefulpartitionedcall_args_2+
'dense_28_statefulpartitionedcall_args_1+
'dense_28_statefulpartitionedcall_args_2+
'dense_29_statefulpartitionedcall_args_1+
'dense_29_statefulpartitionedcall_args_2
identity?? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall? dense_26/StatefulPartitionedCall? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall?
 dense_20/StatefulPartitionedCallStatefulPartitionedCallinputs'dense_20_statefulpartitionedcall_args_1'dense_20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_12031**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-12037*'
_output_shapes
:?????????d?
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0'dense_21_statefulpartitionedcall_args_1'dense_21_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-12065*
Tin
2*'
_output_shapes
:?????????d*
Tout
2*L
fGRE
C__inference_dense_21_layer_call_and_return_conditional_losses_12059?
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0'dense_22_statefulpartitionedcall_args_1'dense_22_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12093*L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_12087*'
_output_shapes
:?????????d*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0'dense_23_statefulpartitionedcall_args_1'dense_23_statefulpartitionedcall_args_2*L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_12115*,
_gradient_op_typePartitionedCall-12121*
Tout
2*
Tin
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????d?
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0'dense_24_statefulpartitionedcall_args_1'dense_24_statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:?????????d*
Tout
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-12149*L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_12143?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0'dense_25_statefulpartitionedcall_args_1'dense_25_statefulpartitionedcall_args_2*
Tin
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????d*
Tout
2*L
fGRE
C__inference_dense_25_layer_call_and_return_conditional_losses_12171*,
_gradient_op_typePartitionedCall-12177?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0'dense_26_statefulpartitionedcall_args_1'dense_26_statefulpartitionedcall_args_2*L
fGRE
C__inference_dense_26_layer_call_and_return_conditional_losses_12199*
Tin
2*'
_output_shapes
:?????????d*,
_gradient_op_typePartitionedCall-12205**
config_proto

GPU 

CPU2J 8*
Tout
2?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0'dense_27_statefulpartitionedcall_args_1'dense_27_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12233*
Tout
2*
Tin
2*L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_12227*'
_output_shapes
:?????????d**
config_proto

GPU 

CPU2J 8?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0'dense_28_statefulpartitionedcall_args_1'dense_28_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12261**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_28_layer_call_and_return_conditional_losses_12255*
Tout
2*'
_output_shapes
:?????????d*
Tin
2?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0'dense_29_statefulpartitionedcall_args_1'dense_29_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12288**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????*L
fGRE
C__inference_dense_29_layer_call_and_return_conditional_losses_12282*
Tout
2*
Tin
2?
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*v
_input_shapese
c:?????????::::::::::::::::::::2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : 
?
?
(__inference_dense_21_layer_call_fn_12721

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*L
fGRE
C__inference_dense_21_layer_call_and_return_conditional_losses_12059*
Tin
2*,
_gradient_op_typePartitionedCall-12065*'
_output_shapes
:?????????d**
config_proto

GPU 

CPU2J 8*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?	
?
C__inference_dense_23_layer_call_and_return_conditional_losses_12750

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:ddi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0N
EluEluBiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*.
_input_shapes
:?????????d::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?	
?
C__inference_dense_25_layer_call_and_return_conditional_losses_12786

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:ddi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dv
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0N
EluEluBiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????d*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
C__inference_dense_20_layer_call_and_return_conditional_losses_12031

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:di
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dv
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????d*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?	
?
C__inference_dense_21_layer_call_and_return_conditional_losses_12059

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:ddi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dv
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0N
EluEluBiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????d*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????d::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
,__inference_sequential_2_layer_call_fn_12397
dense_20_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_20_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20*,
_gradient_op_typePartitionedCall-12374* 
Tin
2*
Tout
2*P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_12373**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*v
_input_shapese
c:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:
 : : : : : : : : : : :. *
(
_user_specified_namedense_20_input: : : : : : : : :	 
?
?
(__inference_dense_23_layer_call_fn_12757

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????d*
Tin
2*
Tout
2*L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_12115*,
_gradient_op_typePartitionedCall-12121?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????d*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?	
?
C__inference_dense_28_layer_call_and_return_conditional_losses_12255

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0N
EluEluBiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????d*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????d::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: : :& "
 
_user_specified_nameinputs
?
?
(__inference_dense_26_layer_call_fn_12811

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*'
_output_shapes
:?????????d*
Tin
2*
Tout
2*L
fGRE
C__inference_dense_26_layer_call_and_return_conditional_losses_12199*,
_gradient_op_typePartitionedCall-12205**
config_proto

GPU 

CPU2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????d*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs
??
?"
!__inference__traced_restore_13304
file_prefix$
 assignvariableop_dense_20_kernel$
 assignvariableop_1_dense_20_bias&
"assignvariableop_2_dense_21_kernel$
 assignvariableop_3_dense_21_bias&
"assignvariableop_4_dense_22_kernel$
 assignvariableop_5_dense_22_bias&
"assignvariableop_6_dense_23_kernel$
 assignvariableop_7_dense_23_bias&
"assignvariableop_8_dense_24_kernel$
 assignvariableop_9_dense_24_bias'
#assignvariableop_10_dense_25_kernel%
!assignvariableop_11_dense_25_bias'
#assignvariableop_12_dense_26_kernel%
!assignvariableop_13_dense_26_bias'
#assignvariableop_14_dense_27_kernel%
!assignvariableop_15_dense_27_bias'
#assignvariableop_16_dense_28_kernel%
!assignvariableop_17_dense_28_bias'
#assignvariableop_18_dense_29_kernel%
!assignvariableop_19_dense_29_bias!
assignvariableop_20_adam_iter#
assignvariableop_21_adam_beta_1#
assignvariableop_22_adam_beta_2"
assignvariableop_23_adam_decay*
&assignvariableop_24_adam_learning_rate
assignvariableop_25_total
assignvariableop_26_count.
*assignvariableop_27_adam_dense_20_kernel_m,
(assignvariableop_28_adam_dense_20_bias_m.
*assignvariableop_29_adam_dense_21_kernel_m,
(assignvariableop_30_adam_dense_21_bias_m.
*assignvariableop_31_adam_dense_22_kernel_m,
(assignvariableop_32_adam_dense_22_bias_m.
*assignvariableop_33_adam_dense_23_kernel_m,
(assignvariableop_34_adam_dense_23_bias_m.
*assignvariableop_35_adam_dense_24_kernel_m,
(assignvariableop_36_adam_dense_24_bias_m.
*assignvariableop_37_adam_dense_25_kernel_m,
(assignvariableop_38_adam_dense_25_bias_m.
*assignvariableop_39_adam_dense_26_kernel_m,
(assignvariableop_40_adam_dense_26_bias_m.
*assignvariableop_41_adam_dense_27_kernel_m,
(assignvariableop_42_adam_dense_27_bias_m.
*assignvariableop_43_adam_dense_28_kernel_m,
(assignvariableop_44_adam_dense_28_bias_m.
*assignvariableop_45_adam_dense_29_kernel_m,
(assignvariableop_46_adam_dense_29_bias_m.
*assignvariableop_47_adam_dense_20_kernel_v,
(assignvariableop_48_adam_dense_20_bias_v.
*assignvariableop_49_adam_dense_21_kernel_v,
(assignvariableop_50_adam_dense_21_bias_v.
*assignvariableop_51_adam_dense_22_kernel_v,
(assignvariableop_52_adam_dense_22_bias_v.
*assignvariableop_53_adam_dense_23_kernel_v,
(assignvariableop_54_adam_dense_23_bias_v.
*assignvariableop_55_adam_dense_24_kernel_v,
(assignvariableop_56_adam_dense_24_bias_v.
*assignvariableop_57_adam_dense_25_kernel_v,
(assignvariableop_58_adam_dense_25_bias_v.
*assignvariableop_59_adam_dense_26_kernel_v,
(assignvariableop_60_adam_dense_26_bias_v.
*assignvariableop_61_adam_dense_27_kernel_v,
(assignvariableop_62_adam_dense_27_bias_v.
*assignvariableop_63_adam_dense_28_kernel_v,
(assignvariableop_64_adam_dense_28_bias_v.
*assignvariableop_65_adam_dense_29_kernel_v,
(assignvariableop_66_adam_dense_29_bias_v
identity_68??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?&
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*?%
value?%B?%CB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:C*?
value?B?CB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Q
dtypesG
E2C	*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
_output_shapes
:*
T0|
AssignVariableOpAssignVariableOp assignvariableop_dense_20_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_20_biasIdentity_1:output:0*
_output_shapes
 *
dtype0N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_21_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_21_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_22_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_22_biasIdentity_5:output:0*
_output_shapes
 *
dtype0N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_23_kernelIdentity_6:output:0*
_output_shapes
 *
dtype0N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_23_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_24_kernelIdentity_8:output:0*
_output_shapes
 *
dtype0N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_24_biasIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_25_kernelIdentity_10:output:0*
_output_shapes
 *
dtype0P
Identity_11IdentityRestoreV2:tensors:11*
_output_shapes
:*
T0?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_25_biasIdentity_11:output:0*
_output_shapes
 *
dtype0P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_26_kernelIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_26_biasIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_27_kernelIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_27_biasIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_28_kernelIdentity_16:output:0*
_output_shapes
 *
dtype0P
Identity_17IdentityRestoreV2:tensors:17*
_output_shapes
:*
T0?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_28_biasIdentity_17:output:0*
_output_shapes
 *
dtype0P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_29_kernelIdentity_18:output:0*
_output_shapes
 *
dtype0P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0?
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_29_biasIdentity_19:output:0*
_output_shapes
 *
dtype0P
Identity_20IdentityRestoreV2:tensors:20*
T0	*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_iterIdentity_20:output:0*
dtype0	*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_beta_1Identity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0?
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_beta_2Identity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_decayIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_learning_rateIdentity_24:output:0*
_output_shapes
 *
dtype0P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:{
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0*
_output_shapes
 *
dtype0P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:{
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_20_kernel_mIdentity_27:output:0*
_output_shapes
 *
dtype0P
Identity_28IdentityRestoreV2:tensors:28*
_output_shapes
:*
T0?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_20_bias_mIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
_output_shapes
:*
T0?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_21_kernel_mIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_21_bias_mIdentity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_22_kernel_mIdentity_31:output:0*
_output_shapes
 *
dtype0P
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_22_bias_mIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_23_kernel_mIdentity_33:output:0*
_output_shapes
 *
dtype0P
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_23_bias_mIdentity_34:output:0*
_output_shapes
 *
dtype0P
Identity_35IdentityRestoreV2:tensors:35*
_output_shapes
:*
T0?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_24_kernel_mIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
_output_shapes
:*
T0?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_24_bias_mIdentity_36:output:0*
_output_shapes
 *
dtype0P
Identity_37IdentityRestoreV2:tensors:37*
_output_shapes
:*
T0?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_25_kernel_mIdentity_37:output:0*
dtype0*
_output_shapes
 P
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_25_bias_mIdentity_38:output:0*
dtype0*
_output_shapes
 P
Identity_39IdentityRestoreV2:tensors:39*
_output_shapes
:*
T0?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_26_kernel_mIdentity_39:output:0*
_output_shapes
 *
dtype0P
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_26_bias_mIdentity_40:output:0*
dtype0*
_output_shapes
 P
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_27_kernel_mIdentity_41:output:0*
dtype0*
_output_shapes
 P
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_27_bias_mIdentity_42:output:0*
_output_shapes
 *
dtype0P
Identity_43IdentityRestoreV2:tensors:43*
_output_shapes
:*
T0?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_28_kernel_mIdentity_43:output:0*
dtype0*
_output_shapes
 P
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_28_bias_mIdentity_44:output:0*
dtype0*
_output_shapes
 P
Identity_45IdentityRestoreV2:tensors:45*
_output_shapes
:*
T0?
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_29_kernel_mIdentity_45:output:0*
_output_shapes
 *
dtype0P
Identity_46IdentityRestoreV2:tensors:46*
_output_shapes
:*
T0?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_29_bias_mIdentity_46:output:0*
dtype0*
_output_shapes
 P
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_20_kernel_vIdentity_47:output:0*
dtype0*
_output_shapes
 P
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_20_bias_vIdentity_48:output:0*
dtype0*
_output_shapes
 P
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_21_kernel_vIdentity_49:output:0*
dtype0*
_output_shapes
 P
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_21_bias_vIdentity_50:output:0*
dtype0*
_output_shapes
 P
Identity_51IdentityRestoreV2:tensors:51*
_output_shapes
:*
T0?
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_22_kernel_vIdentity_51:output:0*
dtype0*
_output_shapes
 P
Identity_52IdentityRestoreV2:tensors:52*
_output_shapes
:*
T0?
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_22_bias_vIdentity_52:output:0*
dtype0*
_output_shapes
 P
Identity_53IdentityRestoreV2:tensors:53*
_output_shapes
:*
T0?
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_23_kernel_vIdentity_53:output:0*
dtype0*
_output_shapes
 P
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_23_bias_vIdentity_54:output:0*
_output_shapes
 *
dtype0P
Identity_55IdentityRestoreV2:tensors:55*
_output_shapes
:*
T0?
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_24_kernel_vIdentity_55:output:0*
dtype0*
_output_shapes
 P
Identity_56IdentityRestoreV2:tensors:56*
_output_shapes
:*
T0?
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_24_bias_vIdentity_56:output:0*
dtype0*
_output_shapes
 P
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_dense_25_kernel_vIdentity_57:output:0*
_output_shapes
 *
dtype0P
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_dense_25_bias_vIdentity_58:output:0*
dtype0*
_output_shapes
 P
Identity_59IdentityRestoreV2:tensors:59*
_output_shapes
:*
T0?
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_dense_26_kernel_vIdentity_59:output:0*
dtype0*
_output_shapes
 P
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_dense_26_bias_vIdentity_60:output:0*
_output_shapes
 *
dtype0P
Identity_61IdentityRestoreV2:tensors:61*
_output_shapes
:*
T0?
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_dense_27_kernel_vIdentity_61:output:0*
_output_shapes
 *
dtype0P
Identity_62IdentityRestoreV2:tensors:62*
_output_shapes
:*
T0?
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_dense_27_bias_vIdentity_62:output:0*
_output_shapes
 *
dtype0P
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_dense_28_kernel_vIdentity_63:output:0*
dtype0*
_output_shapes
 P
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_dense_28_bias_vIdentity_64:output:0*
dtype0*
_output_shapes
 P
Identity_65IdentityRestoreV2:tensors:65*
_output_shapes
:*
T0?
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_dense_29_kernel_vIdentity_65:output:0*
_output_shapes
 *
dtype0P
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_dense_29_bias_vIdentity_66:output:0*
dtype0*
_output_shapes
 ?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHt
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_67Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ?
Identity_68IdentityIdentity_67:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_68Identity_68:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_32AssignVariableOp_322$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_39AssignVariableOp_392*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_49AssignVariableOp_492*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_59AssignVariableOp_592*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :6 :7 :8 :9 :: :; :< := :> :? :@ :A :B :C 
?
?
(__inference_dense_22_layer_call_fn_12739

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*'
_output_shapes
:?????????d*L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_12087*,
_gradient_op_typePartitionedCall-12093*
Tin
2*
Tout
2**
config_proto

GPU 

CPU2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?	
?
C__inference_dense_28_layer_call_and_return_conditional_losses_12840

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dN
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????d*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?o
?
 __inference__wrapped_model_12015
dense_20_input8
4sequential_2_dense_20_matmul_readvariableop_resource9
5sequential_2_dense_20_biasadd_readvariableop_resource8
4sequential_2_dense_21_matmul_readvariableop_resource9
5sequential_2_dense_21_biasadd_readvariableop_resource8
4sequential_2_dense_22_matmul_readvariableop_resource9
5sequential_2_dense_22_biasadd_readvariableop_resource8
4sequential_2_dense_23_matmul_readvariableop_resource9
5sequential_2_dense_23_biasadd_readvariableop_resource8
4sequential_2_dense_24_matmul_readvariableop_resource9
5sequential_2_dense_24_biasadd_readvariableop_resource8
4sequential_2_dense_25_matmul_readvariableop_resource9
5sequential_2_dense_25_biasadd_readvariableop_resource8
4sequential_2_dense_26_matmul_readvariableop_resource9
5sequential_2_dense_26_biasadd_readvariableop_resource8
4sequential_2_dense_27_matmul_readvariableop_resource9
5sequential_2_dense_27_biasadd_readvariableop_resource8
4sequential_2_dense_28_matmul_readvariableop_resource9
5sequential_2_dense_28_biasadd_readvariableop_resource8
4sequential_2_dense_29_matmul_readvariableop_resource9
5sequential_2_dense_29_biasadd_readvariableop_resource
identity??,sequential_2/dense_20/BiasAdd/ReadVariableOp?+sequential_2/dense_20/MatMul/ReadVariableOp?,sequential_2/dense_21/BiasAdd/ReadVariableOp?+sequential_2/dense_21/MatMul/ReadVariableOp?,sequential_2/dense_22/BiasAdd/ReadVariableOp?+sequential_2/dense_22/MatMul/ReadVariableOp?,sequential_2/dense_23/BiasAdd/ReadVariableOp?+sequential_2/dense_23/MatMul/ReadVariableOp?,sequential_2/dense_24/BiasAdd/ReadVariableOp?+sequential_2/dense_24/MatMul/ReadVariableOp?,sequential_2/dense_25/BiasAdd/ReadVariableOp?+sequential_2/dense_25/MatMul/ReadVariableOp?,sequential_2/dense_26/BiasAdd/ReadVariableOp?+sequential_2/dense_26/MatMul/ReadVariableOp?,sequential_2/dense_27/BiasAdd/ReadVariableOp?+sequential_2/dense_27/MatMul/ReadVariableOp?,sequential_2/dense_28/BiasAdd/ReadVariableOp?+sequential_2/dense_28/MatMul/ReadVariableOp?,sequential_2/dense_29/BiasAdd/ReadVariableOp?+sequential_2/dense_29/MatMul/ReadVariableOp?
+sequential_2/dense_20/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_20_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:d?
sequential_2/dense_20/MatMulMatMuldense_20_input3sequential_2/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
,sequential_2/dense_20/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_20_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0?
sequential_2/dense_20/BiasAddBiasAdd&sequential_2/dense_20/MatMul:product:04sequential_2/dense_20/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
+sequential_2/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_21_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:dd*
dtype0?
sequential_2/dense_21/MatMulMatMul&sequential_2/dense_20/BiasAdd:output:03sequential_2/dense_21/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
,sequential_2/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_21_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0?
sequential_2/dense_21/BiasAddBiasAdd&sequential_2/dense_21/MatMul:product:04sequential_2/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dz
sequential_2/dense_21/EluElu&sequential_2/dense_21/BiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
+sequential_2/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_22_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:dd*
dtype0?
sequential_2/dense_22/MatMulMatMul'sequential_2/dense_21/Elu:activations:03sequential_2/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
,sequential_2/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_22_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:d?
sequential_2/dense_22/BiasAddBiasAdd&sequential_2/dense_22/MatMul:product:04sequential_2/dense_22/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0z
sequential_2/dense_22/EluElu&sequential_2/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
+sequential_2/dense_23/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_23_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dd?
sequential_2/dense_23/MatMulMatMul'sequential_2/dense_22/Elu:activations:03sequential_2/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
,sequential_2/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_23_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:d?
sequential_2/dense_23/BiasAddBiasAdd&sequential_2/dense_23/MatMul:product:04sequential_2/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dz
sequential_2/dense_23/EluElu&sequential_2/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
+sequential_2/dense_24/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_24_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:dd*
dtype0?
sequential_2/dense_24/MatMulMatMul'sequential_2/dense_23/Elu:activations:03sequential_2/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
,sequential_2/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_24_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0?
sequential_2/dense_24/BiasAddBiasAdd&sequential_2/dense_24/MatMul:product:04sequential_2/dense_24/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0z
sequential_2/dense_24/EluElu&sequential_2/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
+sequential_2/dense_25/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_25_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dd?
sequential_2/dense_25/MatMulMatMul'sequential_2/dense_24/Elu:activations:03sequential_2/dense_25/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
,sequential_2/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_25_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0?
sequential_2/dense_25/BiasAddBiasAdd&sequential_2/dense_25/MatMul:product:04sequential_2/dense_25/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0z
sequential_2/dense_25/EluElu&sequential_2/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
+sequential_2/dense_26/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_26_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dd?
sequential_2/dense_26/MatMulMatMul'sequential_2/dense_25/Elu:activations:03sequential_2/dense_26/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
,sequential_2/dense_26/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_26_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0?
sequential_2/dense_26/BiasAddBiasAdd&sequential_2/dense_26/MatMul:product:04sequential_2/dense_26/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0z
sequential_2/dense_26/EluElu&sequential_2/dense_26/BiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
+sequential_2/dense_27/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_27_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dd?
sequential_2/dense_27/MatMulMatMul'sequential_2/dense_26/Elu:activations:03sequential_2/dense_27/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
,sequential_2/dense_27/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_27_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0?
sequential_2/dense_27/BiasAddBiasAdd&sequential_2/dense_27/MatMul:product:04sequential_2/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dz
sequential_2/dense_27/EluElu&sequential_2/dense_27/BiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
+sequential_2/dense_28/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_28_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dd?
sequential_2/dense_28/MatMulMatMul'sequential_2/dense_27/Elu:activations:03sequential_2/dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
,sequential_2/dense_28/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_28_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:d?
sequential_2/dense_28/BiasAddBiasAdd&sequential_2/dense_28/MatMul:product:04sequential_2/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dz
sequential_2/dense_28/EluElu&sequential_2/dense_28/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
+sequential_2/dense_29/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_29_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:d*
dtype0?
sequential_2/dense_29/MatMulMatMul'sequential_2/dense_28/Elu:activations:03sequential_2/dense_29/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
,sequential_2/dense_29/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_29_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
sequential_2/dense_29/BiasAddBiasAdd&sequential_2/dense_29/MatMul:product:04sequential_2/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
IdentityIdentity&sequential_2/dense_29/BiasAdd:output:0-^sequential_2/dense_20/BiasAdd/ReadVariableOp,^sequential_2/dense_20/MatMul/ReadVariableOp-^sequential_2/dense_21/BiasAdd/ReadVariableOp,^sequential_2/dense_21/MatMul/ReadVariableOp-^sequential_2/dense_22/BiasAdd/ReadVariableOp,^sequential_2/dense_22/MatMul/ReadVariableOp-^sequential_2/dense_23/BiasAdd/ReadVariableOp,^sequential_2/dense_23/MatMul/ReadVariableOp-^sequential_2/dense_24/BiasAdd/ReadVariableOp,^sequential_2/dense_24/MatMul/ReadVariableOp-^sequential_2/dense_25/BiasAdd/ReadVariableOp,^sequential_2/dense_25/MatMul/ReadVariableOp-^sequential_2/dense_26/BiasAdd/ReadVariableOp,^sequential_2/dense_26/MatMul/ReadVariableOp-^sequential_2/dense_27/BiasAdd/ReadVariableOp,^sequential_2/dense_27/MatMul/ReadVariableOp-^sequential_2/dense_28/BiasAdd/ReadVariableOp,^sequential_2/dense_28/MatMul/ReadVariableOp-^sequential_2/dense_29/BiasAdd/ReadVariableOp,^sequential_2/dense_29/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*v
_input_shapese
c:?????????::::::::::::::::::::2\
,sequential_2/dense_24/BiasAdd/ReadVariableOp,sequential_2/dense_24/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_24/MatMul/ReadVariableOp+sequential_2/dense_24/MatMul/ReadVariableOp2\
,sequential_2/dense_29/BiasAdd/ReadVariableOp,sequential_2/dense_29/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_28/MatMul/ReadVariableOp+sequential_2/dense_28/MatMul/ReadVariableOp2Z
+sequential_2/dense_21/MatMul/ReadVariableOp+sequential_2/dense_21/MatMul/ReadVariableOp2\
,sequential_2/dense_22/BiasAdd/ReadVariableOp,sequential_2/dense_22/BiasAdd/ReadVariableOp2\
,sequential_2/dense_27/BiasAdd/ReadVariableOp,sequential_2/dense_27/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_25/MatMul/ReadVariableOp+sequential_2/dense_25/MatMul/ReadVariableOp2\
,sequential_2/dense_20/BiasAdd/ReadVariableOp,sequential_2/dense_20/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_29/MatMul/ReadVariableOp+sequential_2/dense_29/MatMul/ReadVariableOp2\
,sequential_2/dense_25/BiasAdd/ReadVariableOp,sequential_2/dense_25/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_22/MatMul/ReadVariableOp+sequential_2/dense_22/MatMul/ReadVariableOp2Z
+sequential_2/dense_26/MatMul/ReadVariableOp+sequential_2/dense_26/MatMul/ReadVariableOp2\
,sequential_2/dense_23/BiasAdd/ReadVariableOp,sequential_2/dense_23/BiasAdd/ReadVariableOp2\
,sequential_2/dense_28/BiasAdd/ReadVariableOp,sequential_2/dense_28/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_23/MatMul/ReadVariableOp+sequential_2/dense_23/MatMul/ReadVariableOp2\
,sequential_2/dense_21/BiasAdd/ReadVariableOp,sequential_2/dense_21/BiasAdd/ReadVariableOp2\
,sequential_2/dense_26/BiasAdd/ReadVariableOp,sequential_2/dense_26/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_27/MatMul/ReadVariableOp+sequential_2/dense_27/MatMul/ReadVariableOp2Z
+sequential_2/dense_20/MatMul/ReadVariableOp+sequential_2/dense_20/MatMul/ReadVariableOp:. *
(
_user_specified_namedense_20_input: : : : : : : : :	 :
 : : : : : : : : : : 
?	
?
C__inference_dense_21_layer_call_and_return_conditional_losses_12714

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:ddi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dN
EluEluBiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????d*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?7
?

G__inference_sequential_2_layer_call_and_return_conditional_losses_12373

inputs+
'dense_20_statefulpartitionedcall_args_1+
'dense_20_statefulpartitionedcall_args_2+
'dense_21_statefulpartitionedcall_args_1+
'dense_21_statefulpartitionedcall_args_2+
'dense_22_statefulpartitionedcall_args_1+
'dense_22_statefulpartitionedcall_args_2+
'dense_23_statefulpartitionedcall_args_1+
'dense_23_statefulpartitionedcall_args_2+
'dense_24_statefulpartitionedcall_args_1+
'dense_24_statefulpartitionedcall_args_2+
'dense_25_statefulpartitionedcall_args_1+
'dense_25_statefulpartitionedcall_args_2+
'dense_26_statefulpartitionedcall_args_1+
'dense_26_statefulpartitionedcall_args_2+
'dense_27_statefulpartitionedcall_args_1+
'dense_27_statefulpartitionedcall_args_2+
'dense_28_statefulpartitionedcall_args_1+
'dense_28_statefulpartitionedcall_args_2+
'dense_29_statefulpartitionedcall_args_1+
'dense_29_statefulpartitionedcall_args_2
identity?? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall? dense_26/StatefulPartitionedCall? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall?
 dense_20/StatefulPartitionedCallStatefulPartitionedCallinputs'dense_20_statefulpartitionedcall_args_1'dense_20_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12037**
config_proto

GPU 

CPU2J 8*
Tin
2*
Tout
2*L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_12031*'
_output_shapes
:?????????d?
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0'dense_21_statefulpartitionedcall_args_1'dense_21_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-12065*L
fGRE
C__inference_dense_21_layer_call_and_return_conditional_losses_12059*
Tin
2*'
_output_shapes
:?????????d*
Tout
2?
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0'dense_22_statefulpartitionedcall_args_1'dense_22_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12093*'
_output_shapes
:?????????d*
Tout
2*L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_12087*
Tin
2**
config_proto

GPU 

CPU2J 8?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0'dense_23_statefulpartitionedcall_args_1'dense_23_statefulpartitionedcall_args_2*L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_12115**
config_proto

GPU 

CPU2J 8*
Tout
2*,
_gradient_op_typePartitionedCall-12121*'
_output_shapes
:?????????d*
Tin
2?
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0'dense_24_statefulpartitionedcall_args_1'dense_24_statefulpartitionedcall_args_2*L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_12143**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-12149*
Tin
2*
Tout
2*'
_output_shapes
:?????????d?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0'dense_25_statefulpartitionedcall_args_1'dense_25_statefulpartitionedcall_args_2*
Tin
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-12177*L
fGRE
C__inference_dense_25_layer_call_and_return_conditional_losses_12171*
Tout
2*'
_output_shapes
:?????????d?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0'dense_26_statefulpartitionedcall_args_1'dense_26_statefulpartitionedcall_args_2*
Tin
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-12205*L
fGRE
C__inference_dense_26_layer_call_and_return_conditional_losses_12199*
Tout
2*'
_output_shapes
:?????????d?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0'dense_27_statefulpartitionedcall_args_1'dense_27_statefulpartitionedcall_args_2*'
_output_shapes
:?????????d*
Tin
2*,
_gradient_op_typePartitionedCall-12233*
Tout
2**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_12227?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0'dense_28_statefulpartitionedcall_args_1'dense_28_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-12261*'
_output_shapes
:?????????d*L
fGRE
C__inference_dense_28_layer_call_and_return_conditional_losses_12255*
Tin
2*
Tout
2?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0'dense_29_statefulpartitionedcall_args_1'dense_29_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12288*'
_output_shapes
:?????????*
Tin
2*L
fGRE
C__inference_dense_29_layer_call_and_return_conditional_losses_12282*
Tout
2**
config_proto

GPU 

CPU2J 8?
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*v
_input_shapese
c:?????????::::::::::::::::::::2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : 
?
?
(__inference_dense_24_layer_call_fn_12775

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*'
_output_shapes
:?????????d*L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_12143**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-12149?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?Y
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_12564

inputs+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource+
'dense_21_matmul_readvariableop_resource,
(dense_21_biasadd_readvariableop_resource+
'dense_22_matmul_readvariableop_resource,
(dense_22_biasadd_readvariableop_resource+
'dense_23_matmul_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource+
'dense_24_matmul_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource+
'dense_25_matmul_readvariableop_resource,
(dense_25_biasadd_readvariableop_resource+
'dense_26_matmul_readvariableop_resource,
(dense_26_biasadd_readvariableop_resource+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource+
'dense_28_matmul_readvariableop_resource,
(dense_28_biasadd_readvariableop_resource+
'dense_29_matmul_readvariableop_resource,
(dense_29_biasadd_readvariableop_resource
identity??dense_20/BiasAdd/ReadVariableOp?dense_20/MatMul/ReadVariableOp?dense_21/BiasAdd/ReadVariableOp?dense_21/MatMul/ReadVariableOp?dense_22/BiasAdd/ReadVariableOp?dense_22/MatMul/ReadVariableOp?dense_23/BiasAdd/ReadVariableOp?dense_23/MatMul/ReadVariableOp?dense_24/BiasAdd/ReadVariableOp?dense_24/MatMul/ReadVariableOp?dense_25/BiasAdd/ReadVariableOp?dense_25/MatMul/ReadVariableOp?dense_26/BiasAdd/ReadVariableOp?dense_26/MatMul/ReadVariableOp?dense_27/BiasAdd/ReadVariableOp?dense_27/MatMul/ReadVariableOp?dense_28/BiasAdd/ReadVariableOp?dense_28/MatMul/ReadVariableOp?dense_29/BiasAdd/ReadVariableOp?dense_29/MatMul/ReadVariableOp?
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:d*
dtype0{
dense_20/MatMulMatMulinputs&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:d?
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dd?
dense_21/MatMulMatMuldense_20/BiasAdd:output:0&dense_21/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:d?
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0`
dense_21/EluEludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dd?
dense_22/MatMulMatMuldense_21/Elu:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:d?
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d`
dense_22/EluEludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:dd*
dtype0?
dense_23/MatMulMatMuldense_22/Elu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0?
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0`
dense_23/EluEludense_23/BiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:dd*
dtype0?
dense_24/MatMulMatMuldense_23/Elu:activations:0&dense_24/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0?
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d`
dense_24/EluEludense_24/BiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:dd*
dtype0?
dense_25/MatMulMatMuldense_24/Elu:activations:0&dense_25/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0?
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0`
dense_25/EluEludense_25/BiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dd?
dense_26/MatMulMatMuldense_25/Elu:activations:0&dense_26/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0?
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0`
dense_26/EluEludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dd?
dense_27/MatMulMatMuldense_26/Elu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:d?
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d`
dense_27/EluEludense_27/BiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dd?
dense_28/MatMulMatMuldense_27/Elu:activations:0&dense_28/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0?
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d`
dense_28/EluEludense_28/BiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:d*
dtype0?
dense_29/MatMulMatMuldense_28/Elu:activations:0&dense_29/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
IdentityIdentitydense_29/BiasAdd:output:0 ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*v
_input_shapese
c:?????????::::::::::::::::::::2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : 
?	
?
C__inference_dense_24_layer_call_and_return_conditional_losses_12143

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:ddi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dN
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????d*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : :& "
 
_user_specified_nameinputs
?	
?
C__inference_dense_26_layer_call_and_return_conditional_losses_12804

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dN
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????d*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????d::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?t
?
__inference__traced_save_13090
file_prefix.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableop.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableop.
*savev2_dense_27_kernel_read_readvariableop,
(savev2_dense_27_bias_read_readvariableop.
*savev2_dense_28_kernel_read_readvariableop,
(savev2_dense_28_bias_read_readvariableop.
*savev2_dense_29_kernel_read_readvariableop,
(savev2_dense_29_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_20_kernel_m_read_readvariableop3
/savev2_adam_dense_20_bias_m_read_readvariableop5
1savev2_adam_dense_21_kernel_m_read_readvariableop3
/savev2_adam_dense_21_bias_m_read_readvariableop5
1savev2_adam_dense_22_kernel_m_read_readvariableop3
/savev2_adam_dense_22_bias_m_read_readvariableop5
1savev2_adam_dense_23_kernel_m_read_readvariableop3
/savev2_adam_dense_23_bias_m_read_readvariableop5
1savev2_adam_dense_24_kernel_m_read_readvariableop3
/savev2_adam_dense_24_bias_m_read_readvariableop5
1savev2_adam_dense_25_kernel_m_read_readvariableop3
/savev2_adam_dense_25_bias_m_read_readvariableop5
1savev2_adam_dense_26_kernel_m_read_readvariableop3
/savev2_adam_dense_26_bias_m_read_readvariableop5
1savev2_adam_dense_27_kernel_m_read_readvariableop3
/savev2_adam_dense_27_bias_m_read_readvariableop5
1savev2_adam_dense_28_kernel_m_read_readvariableop3
/savev2_adam_dense_28_bias_m_read_readvariableop5
1savev2_adam_dense_29_kernel_m_read_readvariableop3
/savev2_adam_dense_29_bias_m_read_readvariableop5
1savev2_adam_dense_20_kernel_v_read_readvariableop3
/savev2_adam_dense_20_bias_v_read_readvariableop5
1savev2_adam_dense_21_kernel_v_read_readvariableop3
/savev2_adam_dense_21_bias_v_read_readvariableop5
1savev2_adam_dense_22_kernel_v_read_readvariableop3
/savev2_adam_dense_22_bias_v_read_readvariableop5
1savev2_adam_dense_23_kernel_v_read_readvariableop3
/savev2_adam_dense_23_bias_v_read_readvariableop5
1savev2_adam_dense_24_kernel_v_read_readvariableop3
/savev2_adam_dense_24_bias_v_read_readvariableop5
1savev2_adam_dense_25_kernel_v_read_readvariableop3
/savev2_adam_dense_25_bias_v_read_readvariableop5
1savev2_adam_dense_26_kernel_v_read_readvariableop3
/savev2_adam_dense_26_bias_v_read_readvariableop5
1savev2_adam_dense_27_kernel_v_read_readvariableop3
/savev2_adam_dense_27_bias_v_read_readvariableop5
1savev2_adam_dense_28_kernel_v_read_readvariableop3
/savev2_adam_dense_28_bias_v_read_readvariableop5
1savev2_adam_dense_29_kernel_v_read_readvariableop3
/savev2_adam_dense_29_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*<
value3B1 B+_temp_383376c97529468aaef4ba33c3c0c15a/part*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
dtype0*
_output_shapes
: *
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?&
SaveV2/tensor_namesConst"/device:CPU:0*?%
value?%B?%CB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:C*
dtype0?
SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*?
value?B?CB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:C?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop*savev2_dense_28_kernel_read_readvariableop(savev2_dense_28_bias_read_readvariableop*savev2_dense_29_kernel_read_readvariableop(savev2_dense_29_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_20_kernel_m_read_readvariableop/savev2_adam_dense_20_bias_m_read_readvariableop1savev2_adam_dense_21_kernel_m_read_readvariableop/savev2_adam_dense_21_bias_m_read_readvariableop1savev2_adam_dense_22_kernel_m_read_readvariableop/savev2_adam_dense_22_bias_m_read_readvariableop1savev2_adam_dense_23_kernel_m_read_readvariableop/savev2_adam_dense_23_bias_m_read_readvariableop1savev2_adam_dense_24_kernel_m_read_readvariableop/savev2_adam_dense_24_bias_m_read_readvariableop1savev2_adam_dense_25_kernel_m_read_readvariableop/savev2_adam_dense_25_bias_m_read_readvariableop1savev2_adam_dense_26_kernel_m_read_readvariableop/savev2_adam_dense_26_bias_m_read_readvariableop1savev2_adam_dense_27_kernel_m_read_readvariableop/savev2_adam_dense_27_bias_m_read_readvariableop1savev2_adam_dense_28_kernel_m_read_readvariableop/savev2_adam_dense_28_bias_m_read_readvariableop1savev2_adam_dense_29_kernel_m_read_readvariableop/savev2_adam_dense_29_bias_m_read_readvariableop1savev2_adam_dense_20_kernel_v_read_readvariableop/savev2_adam_dense_20_bias_v_read_readvariableop1savev2_adam_dense_21_kernel_v_read_readvariableop/savev2_adam_dense_21_bias_v_read_readvariableop1savev2_adam_dense_22_kernel_v_read_readvariableop/savev2_adam_dense_22_bias_v_read_readvariableop1savev2_adam_dense_23_kernel_v_read_readvariableop/savev2_adam_dense_23_bias_v_read_readvariableop1savev2_adam_dense_24_kernel_v_read_readvariableop/savev2_adam_dense_24_bias_v_read_readvariableop1savev2_adam_dense_25_kernel_v_read_readvariableop/savev2_adam_dense_25_bias_v_read_readvariableop1savev2_adam_dense_26_kernel_v_read_readvariableop/savev2_adam_dense_26_bias_v_read_readvariableop1savev2_adam_dense_27_kernel_v_read_readvariableop/savev2_adam_dense_27_bias_v_read_readvariableop1savev2_adam_dense_28_kernel_v_read_readvariableop/savev2_adam_dense_28_bias_v_read_readvariableop1savev2_adam_dense_29_kernel_v_read_readvariableop/savev2_adam_dense_29_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *Q
dtypesG
E2C	h
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
value	B :*
dtype0?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHq
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B ?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 ?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :d:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:d:: : : : : : : :d:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:d::d:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:d:: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1:6 :7 :8 :9 :: :; :< := :> :? :@ :A :B :C :D :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 
?
?
C__inference_dense_29_layer_call_and_return_conditional_losses_12857

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
,__inference_sequential_2_layer_call_fn_12459
dense_20_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_20_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20*'
_output_shapes
:?????????* 
Tin
2**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_12435*,
_gradient_op_typePartitionedCall-12436*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*v
_input_shapese
c:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : :. *
(
_user_specified_namedense_20_input: : : : : : : : :	 :
 : : : 
?	
?
C__inference_dense_26_layer_call_and_return_conditional_losses_12199

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:ddi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0N
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????d*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : :& "
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_12490
dense_20_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_20_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-12467*)
f$R"
 __inference__wrapped_model_12015* 
Tin
2*
Tout
2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*v
_input_shapese
c:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_namedense_20_input: : : : : : : : :	 :
 : : : : : : : : : : 
?	
?
C__inference_dense_27_layer_call_and_return_conditional_losses_12227

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dN
EluEluBiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
C__inference_dense_20_layer_call_and_return_conditional_losses_12696

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????d*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
(__inference_dense_20_layer_call_fn_12703

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*,
_gradient_op_typePartitionedCall-12037**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_12031*'
_output_shapes
:?????????d*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????d*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
(__inference_dense_27_layer_call_fn_12829

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12233**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????d*
Tout
2*
Tin
2*L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_12227?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????d*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs
?
?
,__inference_sequential_2_layer_call_fn_12661

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20*P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_12373*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-12374* 
Tin
2*
Tout
2**
config_proto

GPU 

CPU2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*v
_input_shapese
c:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : 
?	
?
C__inference_dense_24_layer_call_and_return_conditional_losses_12768

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:ddi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dN
EluEluBiasAdd:output:0*'
_output_shapes
:?????????d*
T0?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*?
serving_default?
I
dense_20_input7
 serving_default_dense_20_input:0?????????<
dense_290
StatefulPartitionedCall:0?????????tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:??
?N
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"?J
_tf_keras_sequential?I{"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_2", "layers": [{"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "batch_input_shape": [null, 13], "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 13}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "batch_input_shape": [null, 13], "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": ["mean_squared_error"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "dense_20_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 13], "config": {"batch_input_shape": [null, 13], "dtype": "float32", "sparse": false, "name": "dense_20_input"}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 13], "config": {"name": "dense_20", "trainable": true, "batch_input_shape": [null, 13], "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 13}}}}
?

kernel
bias
regularization_losses
trainable_variables
 	variables
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}}
?

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}}
?

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}}
?

.kernel
/bias
0regularization_losses
1trainable_variables
2	variables
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}}
?

4kernel
5bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}}
?

:kernel
;bias
<regularization_losses
=trainable_variables
>	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}}
?

@kernel
Abias
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}}
?

Fkernel
Gbias
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}}
?

Lkernel
Mbias
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}}
?
Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratem?m?m?m?"m?#m?(m?)m?.m?/m?4m?5m?:m?;m?@m?Am?Fm?Gm?Lm?Mm?v?v?v?v?"v?#v?(v?)v?.v?/v?4v?5v?:v?;v?@v?Av?Fv?Gv?Lv?Mv?"
	optimizer
 "
trackable_list_wrapper
?
0
1
2
3
"4
#5
(6
)7
.8
/9
410
511
:12
;13
@14
A15
F16
G17
L18
M19"
trackable_list_wrapper
?
0
1
2
3
"4
#5
(6
)7
.8
/9
410
511
:12
;13
@14
A15
F16
G17
L18
M19"
trackable_list_wrapper
?
Wmetrics
regularization_losses
Xnon_trainable_variables
trainable_variables
Ylayer_regularization_losses

Zlayers
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
[metrics
regularization_losses
\non_trainable_variables
trainable_variables
]layer_regularization_losses

^layers
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:d2dense_20/kernel
:d2dense_20/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
_metrics
regularization_losses
`non_trainable_variables
trainable_variables
alayer_regularization_losses

blayers
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:dd2dense_21/kernel
:d2dense_21/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
cmetrics
regularization_losses
dnon_trainable_variables
trainable_variables
elayer_regularization_losses

flayers
 	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:dd2dense_22/kernel
:d2dense_22/bias
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
gmetrics
$regularization_losses
hnon_trainable_variables
%trainable_variables
ilayer_regularization_losses

jlayers
&	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:dd2dense_23/kernel
:d2dense_23/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
kmetrics
*regularization_losses
lnon_trainable_variables
+trainable_variables
mlayer_regularization_losses

nlayers
,	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:dd2dense_24/kernel
:d2dense_24/bias
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
ometrics
0regularization_losses
pnon_trainable_variables
1trainable_variables
qlayer_regularization_losses

rlayers
2	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:dd2dense_25/kernel
:d2dense_25/bias
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?
smetrics
6regularization_losses
tnon_trainable_variables
7trainable_variables
ulayer_regularization_losses

vlayers
8	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:dd2dense_26/kernel
:d2dense_26/bias
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
?
wmetrics
<regularization_losses
xnon_trainable_variables
=trainable_variables
ylayer_regularization_losses

zlayers
>	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:dd2dense_27/kernel
:d2dense_27/bias
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
?
{metrics
Bregularization_losses
|non_trainable_variables
Ctrainable_variables
}layer_regularization_losses

~layers
D	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:dd2dense_28/kernel
:d2dense_28/bias
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
?
metrics
Hregularization_losses
?non_trainable_variables
Itrainable_variables
 ?layer_regularization_losses
?layers
J	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:d2dense_29/kernel
:2dense_29/bias
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
?
?metrics
Nregularization_losses
?non_trainable_variables
Otrainable_variables
 ?layer_regularization_losses
?layers
P	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
	7

8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?
_fn_kwargs
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "mean_squared_error", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mean_squared_error", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
&:$d2Adam/dense_20/kernel/m
 :d2Adam/dense_20/bias/m
&:$dd2Adam/dense_21/kernel/m
 :d2Adam/dense_21/bias/m
&:$dd2Adam/dense_22/kernel/m
 :d2Adam/dense_22/bias/m
&:$dd2Adam/dense_23/kernel/m
 :d2Adam/dense_23/bias/m
&:$dd2Adam/dense_24/kernel/m
 :d2Adam/dense_24/bias/m
&:$dd2Adam/dense_25/kernel/m
 :d2Adam/dense_25/bias/m
&:$dd2Adam/dense_26/kernel/m
 :d2Adam/dense_26/bias/m
&:$dd2Adam/dense_27/kernel/m
 :d2Adam/dense_27/bias/m
&:$dd2Adam/dense_28/kernel/m
 :d2Adam/dense_28/bias/m
&:$d2Adam/dense_29/kernel/m
 :2Adam/dense_29/bias/m
&:$d2Adam/dense_20/kernel/v
 :d2Adam/dense_20/bias/v
&:$dd2Adam/dense_21/kernel/v
 :d2Adam/dense_21/bias/v
&:$dd2Adam/dense_22/kernel/v
 :d2Adam/dense_22/bias/v
&:$dd2Adam/dense_23/kernel/v
 :d2Adam/dense_23/bias/v
&:$dd2Adam/dense_24/kernel/v
 :d2Adam/dense_24/bias/v
&:$dd2Adam/dense_25/kernel/v
 :d2Adam/dense_25/bias/v
&:$dd2Adam/dense_26/kernel/v
 :d2Adam/dense_26/bias/v
&:$dd2Adam/dense_27/kernel/v
 :d2Adam/dense_27/bias/v
&:$dd2Adam/dense_28/kernel/v
 :d2Adam/dense_28/bias/v
&:$d2Adam/dense_29/kernel/v
 :2Adam/dense_29/bias/v
?2?
,__inference_sequential_2_layer_call_fn_12661
,__inference_sequential_2_layer_call_fn_12459
,__inference_sequential_2_layer_call_fn_12686
,__inference_sequential_2_layer_call_fn_12397?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_2_layer_call_and_return_conditional_losses_12564
G__inference_sequential_2_layer_call_and_return_conditional_losses_12636
G__inference_sequential_2_layer_call_and_return_conditional_losses_12300
G__inference_sequential_2_layer_call_and_return_conditional_losses_12336?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_12015?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *-?*
(?%
dense_20_input?????????
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
(__inference_dense_20_layer_call_fn_12703?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_20_layer_call_and_return_conditional_losses_12696?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_21_layer_call_fn_12721?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_21_layer_call_and_return_conditional_losses_12714?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_22_layer_call_fn_12739?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_22_layer_call_and_return_conditional_losses_12732?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_23_layer_call_fn_12757?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_23_layer_call_and_return_conditional_losses_12750?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_24_layer_call_fn_12775?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_24_layer_call_and_return_conditional_losses_12768?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_25_layer_call_fn_12793?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_25_layer_call_and_return_conditional_losses_12786?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_26_layer_call_fn_12811?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_26_layer_call_and_return_conditional_losses_12804?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_27_layer_call_fn_12829?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_27_layer_call_and_return_conditional_losses_12822?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_28_layer_call_fn_12847?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_28_layer_call_and_return_conditional_losses_12840?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_29_layer_call_fn_12864?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_29_layer_call_and_return_conditional_losses_12857?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
9B7
#__inference_signature_wrapper_12490dense_20_input
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ?
 __inference__wrapped_model_12015?"#()./45:;@AFGLM7?4
-?*
(?%
dense_20_input?????????
? "3?0
.
dense_29"?
dense_29?????????{
(__inference_dense_27_layer_call_fn_12829O@A/?,
%?"
 ?
inputs?????????d
? "??????????d?
C__inference_dense_28_layer_call_and_return_conditional_losses_12840\FG/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_12636v"#()./45:;@AFGLM7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
,__inference_sequential_2_layer_call_fn_12459q"#()./45:;@AFGLM??<
5?2
(?%
dense_20_input?????????
p 

 
? "??????????{
(__inference_dense_29_layer_call_fn_12864OLM/?,
%?"
 ?
inputs?????????d
? "???????????
C__inference_dense_23_layer_call_and_return_conditional_losses_12750\()/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? {
(__inference_dense_28_layer_call_fn_12847OFG/?,
%?"
 ?
inputs?????????d
? "??????????d?
,__inference_sequential_2_layer_call_fn_12686i"#()./45:;@AFGLM7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
,__inference_sequential_2_layer_call_fn_12661i"#()./45:;@AFGLM7?4
-?*
 ?
inputs?????????
p

 
? "??????????{
(__inference_dense_25_layer_call_fn_12793O45/?,
%?"
 ?
inputs?????????d
? "??????????d{
(__inference_dense_23_layer_call_fn_12757O()/?,
%?"
 ?
inputs?????????d
? "??????????d?
#__inference_signature_wrapper_12490?"#()./45:;@AFGLMI?F
? 
??<
:
dense_20_input(?%
dense_20_input?????????"3?0
.
dense_29"?
dense_29?????????{
(__inference_dense_22_layer_call_fn_12739O"#/?,
%?"
 ?
inputs?????????d
? "??????????d?
C__inference_dense_22_layer_call_and_return_conditional_losses_12732\"#/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? {
(__inference_dense_20_layer_call_fn_12703O/?,
%?"
 ?
inputs?????????
? "??????????d?
G__inference_sequential_2_layer_call_and_return_conditional_losses_12336~"#()./45:;@AFGLM??<
5?2
(?%
dense_20_input?????????
p 

 
? "%?"
?
0?????????
? ?
C__inference_dense_27_layer_call_and_return_conditional_losses_12822\@A/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_12564v"#()./45:;@AFGLM7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
C__inference_dense_26_layer_call_and_return_conditional_losses_12804\:;/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? ?
,__inference_sequential_2_layer_call_fn_12397q"#()./45:;@AFGLM??<
5?2
(?%
dense_20_input?????????
p

 
? "???????????
C__inference_dense_29_layer_call_and_return_conditional_losses_12857\LM/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????
? {
(__inference_dense_24_layer_call_fn_12775O.//?,
%?"
 ?
inputs?????????d
? "??????????d?
C__inference_dense_25_layer_call_and_return_conditional_losses_12786\45/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? ?
C__inference_dense_20_layer_call_and_return_conditional_losses_12696\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????d
? ?
C__inference_dense_21_layer_call_and_return_conditional_losses_12714\/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? {
(__inference_dense_26_layer_call_fn_12811O:;/?,
%?"
 ?
inputs?????????d
? "??????????d{
(__inference_dense_21_layer_call_fn_12721O/?,
%?"
 ?
inputs?????????d
? "??????????d?
C__inference_dense_24_layer_call_and_return_conditional_losses_12768\.//?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_12300~"#()./45:;@AFGLM??<
5?2
(?%
dense_20_input?????????
p

 
? "%?"
?
0?????????
? 