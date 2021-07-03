import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *

def eam_layer():
    class EAM(Layer):
        def __init__(self, filters = 64, name = None, **kwargs):
            super(EAM, self).__init__()
            self.filters = filters 
            if name is not None:
                self._name = name
        def build(self, input_shape):    
            self.conv11 = Conv2D(self.filters, 3, activation =  'relu', dilation_rate = 1, padding = 'same', name = 'C11')
            self.conv12 = Conv2D(self.filters, 3, activation =  'relu', dilation_rate = 2, padding = 'same', name = 'C12')

            self.conv21 = Conv2D(self.filters, 3, activation =  'relu', dilation_rate = 3, padding = 'same', name = 'C21')
            self.conv22 = Conv2D(self.filters, 3, activation =  'relu', dilation_rate = 4, padding = 'same', name = 'C22')

            self.conc = Concatenate(-1, name = 'Concat')

            self.conv31 = Conv2D(self.filters, 3, activation =  'relu', padding = 'same', name = 'C31')

            self.resi_1 = Add(name = 'Residual_1')

            self.conv41 = Conv2D(self.filters, 3, activation =  'relu', padding = 'same', name = 'C41')
            self.conv42 = Conv2D(self.filters, 3, activation =  None, padding = 'same', name = 'C42')

            self.resi_2 = Add(name = 'Residual_2')

            self.conv51 = Conv2D(self.filters, 3, activation =  'relu', padding = 'same', name = 'C51')
            self.conv52 = Conv2D(self.filters, 3, activation =  'relu', padding = 'same', name = 'C52')
            self.conv61 = Conv2D(self.filters, 1, activation =  None, padding = 'same', name = 'C61')

            self.resi_3 = Add(name = 'Residual_3')

            self.glob = GlobalAveragePooling2D(name = 'Global_Pool')

            self.reshape = Reshape((1, 1, self.filters), name = 'Reshape')

            self.conv71 = Conv2D(4, 3, activation =  'relu', padding = 'same', name = 'C71')
            self.conv72 = Conv2D(self.filters, 3, activation =  'sigmoid', padding = 'same', name = 'C72')

            self.mul = Multiply(name = 'Multiply')

            self.resi_4 = Add(name = 'Residual_4')
        def call(self, inp):
            conv11 = self.conv11(inp)
            conv12 = self.conv12(conv11)

            conv21 = self.conv21(inp)
            conv22 = self.conv22(conv21)
            conc = self.conc([conv12, conv22])

            conv31 = self.conv31(conc)
            resi_1 = self.resi_1([conv31, inp])

            conv41 = self.conv41(resi_1)
            conv42 = self.conv42(conv41)
            
            resi_2 = self.resi_2([resi_1, conv42])
            resi_2 = Activation('relu')(resi_2)

            conv51 = self.conv51(resi_2)
            conv52 = self.conv52(conv51)
            conv61 = self.conv61(conv52)

            resi_3 = self.resi_3([resi_2, conv61])
            resi_3 = Activation('relu')(resi_3)

            glob = self.glob(resi_3)

            reshape = self.reshape(glob)

            conv71 = self.conv71(reshape)
            conv72 = self.conv72(conv71)

            mul = self.mul([resi_3, conv72])
            
            resi_4 = self.resi_4([inp, mul])
            return resi_4
        
        def get_config(self): #https://stackoverflow.com/questions/58678836/notimplementederror-layers-with-arguments-in-init-must-override-get-conf/58799021
            config = super().get_config().copy()
            config.update({'filter_size' : self.filters})
            return config
    return EAM()
