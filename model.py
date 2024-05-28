def glfnet(data):
    
    filters = [64, 128, 256, 512, 1024, 512, 256, 128, 64]
    
    blocks = len(filters)

    stochastic_depth_rate = 0.0
    
    image_size = data.shape[1]
    
    input_shape = (data.shape[1], data.shape[2], data.shape[3])
    

    
    class StochasticDepth(layers.Layer):
        def __init__(self, drop_prop, **kwargs):
            super(StochasticDepth, self).__init__(**kwargs)
            self.drop_prob = drop_prop
    
        def call(self, x, training=training):
            if training:
                keep_prob = 1 - self.drop_prob
                shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
                random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
                random_tensor = tf.floor(random_tensor)
                return (x / keep_prob) * random_tensor
            return x
    
    def wide_focus(x, filters, dropout_rate): 
        """
        Wide-Focus module.
        """ 
        x1 = layers.Conv2D(filters, 3, padding='same', activation=tf.nn.gelu)(x)
        x1 = layers.Dropout(0.1)(x1)
        x2 = layers.Conv2D(filters, 3, padding='same', dilation_rate=2, activation=tf.nn.gelu)(x)
        x2 = layers.Dropout(0.1)(x2)
        x3 = layers.Conv2D(filters, 3, padding='same', dilation_rate=3, activation=tf.nn.gelu)(x)
        x3 = layers.Dropout(0.1)(x3)
        added = layers.Add()([x1,x2, x3])
        x_out = layers.Conv2D(filters, 3, padding='same', activation=tf.nn.gelu)(added)
        x_out = layers.Dropout(0.1)(x_out)
        return x_out

    from tensorflow.keras.layers import Layer, Dense, Conv2D, Dropout, MultiHeadAttention, BatchNormalization, \
    DepthwiseConv2D, UpSampling2D
    from tensorflow.keras.models import Sequential
    from tensorflow import Tensor, divide, concat, random, split, reshape, transpose, float32
    from typing import List, Union, Iterable
    
        
    def att(x_in):
# %%
        class TRUELocalFilter(Layer):
            """
            Global Filter Layer
            """
            def __init__(self, dim, h, w):
                    super().__init__()
                    self.dim = dim
                    self.h = h
                    self.w = w

            def build(self, input_shape):
                self.cw1 = self.add_weight(shape=(self.dim, self.h//4, self.w//2//4+1, 2), 
                                                      dtype = tf.float32,
                                                      trainable = True,
                                                      name="cw1"
                                                      )
                self.cw2 = self.add_weight(shape=(self.dim, self.h//4, self.w//2//4+1, 2), 
                                                      dtype = tf.float32,
                                                      trainable = True,
                                                      name="cw2"
                                                      )
                self.cw3 = self.add_weight(shape=(self.dim, self.h//4, self.w//2//4+1, 2), 
                                                      dtype = tf.float32,
                                                      trainable = True,
                                                      name="cw3"
                                                      )
                self.cw4 = self.add_weight(shape=(self.dim, self.h//4, self.w//2//4+1, 2), 
                                                      dtype = tf.float32,
                                                      trainable = True,
                                                      name="cw4"
                                                      )
                self.cw5 = self.add_weight(shape=(self.dim, self.h//4, self.w//2//4+1, 2), 
                                                      dtype = tf.float32,
                                                      trainable = True,
                                                      name="cw5"
                                                      )
                self.cw6 = self.add_weight(shape=(self.dim, self.h//4, self.w//2//4+1, 2), 
                                                      dtype = tf.float32,
                                                      trainable = True,
                                                      name="cw6"
                                                      )
                self.cw7 = self.add_weight(shape=(self.dim, self.h//4, self.w//2//4+1, 2), 
                                                      dtype = tf.float32,
                                                      trainable = True,
                                                      name="cw7"
                                                      )
                self.cw8 = self.add_weight(shape=(self.dim, self.h//4, self.w//2//4+1, 2), 
                                                      dtype = tf.float32,
                                                      trainable = True,
                                                      name="cw8"
                                                      )
                self.cw9 = self.add_weight(shape=(self.dim, self.h//4, self.w//2//4+1, 2), 
                                                      dtype = tf.float32,
                                                      trainable = True,
                                                      name="cw9"
                                                      )
                self.cw10 = self.add_weight(shape=(self.dim, self.h//4, self.w//2//4+1, 2), 
                                                      dtype = tf.float32,
                                                      trainable = True,
                                                      name="cw10"
                                                      )
                self.cw11 = self.add_weight(shape=(self.dim, self.h//4, self.w//2//4+1, 2), 
                                                      dtype = tf.float32,
                                                      trainable = True,
                                                      name="cw11"
                                                      )
                self.cw12 = self.add_weight(shape=(self.dim, self.h//4, self.w//2//4+1, 2), 
                                                      dtype = tf.float32,
                                                      trainable = True,
                                                      name="cw12"
                                                      )
                self.cw13 = self.add_weight(shape=(self.dim, self.h//4, self.w//2//4+1, 2), 
                                                      dtype = tf.float32,
                                                      trainable = True,
                                                      name="cw13"
                                                      )
                self.cw14 = self.add_weight(shape=(self.dim, self.h//4, self.w//2//4+1, 2), 
                                                      dtype = tf.float32,
                                                      trainable = True,
                                                      name="cw14"
                                                      )
                self.cw15 = self.add_weight(shape=(self.dim, self.h//4, self.w//2//4+1, 2), 
                                                      dtype = tf.float32,
                                                      trainable = True,
                                                      name="cw15"
                                                      )
                self.cw16 = self.add_weight(shape=(self.dim, self.h//4, self.w//2//4+1, 2), 
                                                      dtype = tf.float32,
                                                      trainable = True,
                                                      name="cw16"
                                                      )
                
        
            def call(self, x):
                x = tf.cast(x, tf.float32)
                B, a, b, C = x.shape
                x = tf.transpose(x, [0,3,1,2])
                #
                temph = a//4
                tempw = b//4
                x1 = x[:,:,:temph*1, :tempw*1]
                x2 = x[:,:,:temph*1, tempw*1:tempw*2]
                x3 = x[:,:,:temph*1, tempw*2:tempw*3]
                x4 = x[:,:,:temph*1, tempw*3:]
                #
                x5 = x[:,:,temph*1:temph*2, :tempw*1]
                x6 = x[:,:,temph*1:temph*2, tempw*1:tempw*2]
                x7 = x[:,:,temph*1:temph*2, tempw*2:tempw*3]
                x8 = x[:,:,temph*1:temph*2, tempw*3:]
                #
                x9 = x[:,:,temph*2:temph*3, :tempw*1]
                x10 = x[:,:,temph*2:temph*3, tempw*1:tempw*2]
                x11 = x[:,:,temph*2:temph*3, tempw*2:tempw*3]
                x12 = x[:,:,temph*2:temph*3, tempw*3:]
                #
                x13 = x[:,:,temph*3:, :tempw*1]
                x14 = x[:,:,temph*3:, tempw*1:tempw*2]
                x15 = x[:,:,temph*3:, tempw*2:tempw*3]
                x16 = x[:,:,temph*3:, tempw*3:]
                #
                #
                #
                _, _, aa, bb = x1.shape
                x1 = tf.signal.rfft2d(x1, fft_length=[aa, bb])
                x2 = tf.signal.rfft2d(x2, fft_length=[aa, bb])
                x3 = tf.signal.rfft2d(x3, fft_length=[aa, bb])
                x4 = tf.signal.rfft2d(x4, fft_length=[aa, bb])
                x5 = tf.signal.rfft2d(x5, fft_length=[aa, bb])
                x6 = tf.signal.rfft2d(x6, fft_length=[aa, bb])
                x7 = tf.signal.rfft2d(x7, fft_length=[aa, bb])
                x8 = tf.signal.rfft2d(x8, fft_length=[aa, bb])
                x9 = tf.signal.rfft2d(x9, fft_length=[aa, bb])
                x10 = tf.signal.rfft2d(x10, fft_length=[aa, bb])
                x11 = tf.signal.rfft2d(x11, fft_length=[aa, bb])
                x12 = tf.signal.rfft2d(x12, fft_length=[aa, bb])
                x13 = tf.signal.rfft2d(x13, fft_length=[aa, bb])
                x14 = tf.signal.rfft2d(x14, fft_length=[aa, bb])
                x15 = tf.signal.rfft2d(x15, fft_length=[aa, bb])
                x16 = tf.signal.rfft2d(x16, fft_length=[aa, bb])
                
                w1 = self.cw1
                w2 = self.cw2
                w3 = self.cw3
                w4 = self.cw4
                w5 = self.cw5
                w6 = self.cw6
                w7 = self.cw7
                w8 = self.cw8
                w9 = self.cw9
                w10 = self.cw10
                w11 = self.cw11
                w12 = self.cw12
                w13 = self.cw13
                w14 = self.cw14
                w15 = self.cw15
                w16 = self.cw16

                w1 = tf.complex(w1[..., 0], w1[..., 1])
                w2 = tf.complex(w2[..., 0], w2[..., 1])
                w3 = tf.complex(w3[..., 0], w3[..., 1])
                w4 = tf.complex(w4[..., 0], w4[..., 1])
                w5 = tf.complex(w5[..., 0], w5[..., 1])
                w6 = tf.complex(w6[..., 0], w6[..., 1])
                w7 = tf.complex(w7[..., 0], w7[..., 1])
                w8 = tf.complex(w8[..., 0], w8[..., 1])
                w9 = tf.complex(w9[..., 0], w9[..., 1])
                w10 = tf.complex(w10[..., 0], w10[..., 1])
                w11 = tf.complex(w11[..., 0], w11[..., 1])
                w12 = tf.complex(w12[..., 0], w12[..., 1])
                w13 = tf.complex(w13[..., 0], w13[..., 1])
                w14 = tf.complex(w14[..., 0], w14[..., 1])
                w15 = tf.complex(w15[..., 0], w15[..., 1])
                w16 = tf.complex(w16[..., 0], w16[..., 1])

                x1 = x1*w1
                x2 = x2*w2
                x3 = x3*w3
                x4 = x4*w4
                #
                x5 = x5*w5
                x6 = x6*w6
                x7 = x7*w7
                x8 = x8*w8
                #
                x9 = x9*w9
                x10 = x10*w10
                x11 = x11*w11
                x12 = x12*w12
                #
                x13 = x13*w13
                x14 = x14*w14
                x15 = x15*w15
                x16 = x16*w16
                
                x1 = tf.signal.irfft2d(x1, fft_length=[aa,bb])
                x2 = tf.signal.irfft2d(x2, fft_length=[aa,bb])
                x3 = tf.signal.irfft2d(x3, fft_length=[aa,bb])
                x4 = tf.signal.irfft2d(x4, fft_length=[aa,bb])
                x5 = tf.signal.irfft2d(x5, fft_length=[aa,bb])
                x6 = tf.signal.irfft2d(x6, fft_length=[aa,bb])
                x7 = tf.signal.irfft2d(x7, fft_length=[aa,bb])
                x8 = tf.signal.irfft2d(x8, fft_length=[aa,bb])
                x9 = tf.signal.irfft2d(x9, fft_length=[aa,bb])
                x10 = tf.signal.irfft2d(x10, fft_length=[aa,bb])
                x11 = tf.signal.irfft2d(x11, fft_length=[aa,bb])
                x12 = tf.signal.irfft2d(x12, fft_length=[aa,bb])
                x13 = tf.signal.irfft2d(x13, fft_length=[aa,bb])
                x14 = tf.signal.irfft2d(x14, fft_length=[aa,bb])
                x15 = tf.signal.irfft2d(x15, fft_length=[aa,bb])
                x16 = tf.signal.irfft2d(x16, fft_length=[aa,bb])
                #
                x1_1 = Concatenate(axis=-1)([x1,x2,x3,x4])
                x1_2 = Concatenate(axis=-1)([x5,x6,x7,x8])
                x1_3 = Concatenate(axis=-1)([x9,x10,x11,x12])
                x1_4 = Concatenate(axis=-1)([x13,x14,x15,x16])
                #
                x = Concatenate(axis=-2)([x1_1,x1_2,x1_3,x1_4])
                #
                #
                #
                x = tf.transpose(x, [0,2,3,1])
                
                return x
  
        class GlobalFilter(Layer):
            """
            Global Filter Layer
            """
            def __init__(self, dim, h, w):
                    super().__init__()
                    self.dim = dim
                    self.h = h
                    self.w = w

            def build(self, input_shape):
                self.complex_weight = self.add_weight(shape=(self.dim, self.h, self.w//2+1, 2), 
                                                      dtype = tf.float32,
                                                      trainable = True
                                                      )
        
            def call(self, x):
                x = tf.cast(x, tf.float32)
                B, a, b, C = x.shape
                # print("x:------------------->", x.shape)
                x = tf.transpose(x, [0,3,1,2])
                x = tf.signal.rfft2d(x, fft_length=[a, b])
                # print("x:------------------->", x.shape)
                
                weight = self.complex_weight
                weight = tf.complex(weight[..., 0], weight[..., 1])
                x = x * weight
                x = tf.signal.irfft2d(x, fft_length=[a, b])
                x = tf.transpose(x, [0,2,3,1])
                
                return x
 
# %%
        b, h, w, c = x_in.shape
        local_attention_a = TRUELocalFilter(dim=c, h=h, w=w)(x_in)
        global_attention_a = GlobalFilter(dim=c, h=h, w=w)(x_in) 
         
        local_attention_a = Conv2D(x_in.shape[-1], 3, 1, padding="same", activation="relu")(local_attention_a)
        global_attention_a = Conv2D(x_in.shape[-1], 3, 1, padding="same", activation="relu")(global_attention_a)
        
        attention_output = concatenate([local_attention_a, global_attention_a])
                
        attention_output = Conv2D(x_in.shape[-1], 3, 1, padding="same", activation="relu")(attention_output)
        x2 = layers.Add()([attention_output, x_in]) 
        x3 = layers.LayerNormalization(epsilon=1e-5)(x2)
        x3 = wide_focus(x3, filters=c, dropout_rate=0.0)
        x3 = layers.Add()([x3, x2])
                        
        return x3
    
    def create_model(
        image_size=image_size,
        input_shape=input_shape,
    ):
    
        inputs = layers.Input(input_shape)
        
        initializer = 'he_normal'
        drp_out = 0.3
        act = "relu"    
    
        scale_img_2 = layers.AveragePooling2D(2,2)(inputs)
        scale_img_3 = layers.AveragePooling2D(2,2)(scale_img_2)
        scale_img_4 = layers.AveragePooling2D(2,2)(scale_img_3)
       
        # first block
        x1 = layers.LayerNormalization(epsilon=1e-5)(inputs[:,:,:,-1])
        x11 = tf.expand_dims(x1, -1)                                
        x11 = Conv2D(filters[0], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
        x11 = Conv2D(filters[0], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
        x11 = Dropout(drp_out)(x11)
        x11 = MaxPooling2D((2,2))(x11)
        out = att(x11)
        out = att(out)
        skip1=out
        
        # second block
        x1 = layers.LayerNormalization(epsilon=1e-5)(out)
        x11=x1
        x11 = concatenate([Conv2D(filters[0], 3, padding="same", activation=act)(scale_img_2), x11], axis=3)
        x11 = Conv2D(filters[1], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
        x11 = Conv2D(filters[1], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
        x11 = Dropout(drp_out)(x11)
        x11 = MaxPooling2D((2,2))(x11)
        out = att(x11)
        out = att(out)
        skip2=out
        
        # third block
        x1 = layers.LayerNormalization(epsilon=1e-5)(out)
        x11=x1
        x11 = concatenate([Conv2D(filters[1], 3, padding="same", activation=act)(scale_img_3), x11], axis=3)
        x11 = Conv2D(filters[2], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
        x11 = Conv2D(filters[2], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
        x11 = Dropout(drp_out)(x11)
        x11 = MaxPooling2D((2,2))(x11)
        out = att(x11)
        out = att(out)
        skip3=out
        
        # fourth block
        x1 = layers.LayerNormalization(epsilon=1e-5)(out)
        x11=x1
        x11 = concatenate([Conv2D(filters[2], 3, padding="same", activation=act)(scale_img_4), x11], axis=3)
        x11 = Conv2D(filters[3], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
        x11 = Conv2D(filters[3], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
        x11 = Dropout(drp_out)(x11)
        x11 = MaxPooling2D((2,2))(x11)
        out = att(x11)
        out = att(out)
        skip4 = out
         
        # fifth block
        x1 = layers.LayerNormalization(epsilon=1e-5)(out)
        x11=x1
        x11 = Conv2D(filters[4], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
        x11 = Conv2D(filters[4], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
        x11 = Dropout(drp_out)(x11)
        x11 = MaxPooling2D((2,2))(x11)
        out = att(x11)
        out = att(out) 
        
        # sixth block
        x1 = layers.LayerNormalization(epsilon=1e-5)(out)
        x11=x1
        x11 = Conv2D(filters[5], 2, padding="same", activation=act, kernel_initializer=initializer)(UpSampling2D(size=(2,2))(x11))
        x11 = concatenate([skip4,x11], axis=3)
        x11 = Conv2D(filters[5], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
        x11 = Conv2D(filters[5], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
        x11 = Dropout(drp_out)(x11)
        out = att(x11)
        out = att(out)               
        
        # seventh block
        x1 = layers.LayerNormalization(epsilon=1e-5)(out)
        x11=x1
        x11 = Conv2D(filters[6], 2, padding="same", activation=act, kernel_initializer=initializer)(UpSampling2D(size=(2,2))(x11))
        x11 = concatenate([skip3,x11], axis=3)
        x11 = Conv2D(filters[6], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
        x11 = Conv2D(filters[6], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
        x11 = Dropout(drp_out)(x11)
        out = att(x11)
        out = att(out)
        skip7=out
        
        # eighth block
        x1 = layers.LayerNormalization(epsilon=1e-5)(out)
        x11=x1
        x11 = Conv2D(filters[7], 2, padding="same", activation=act, kernel_initializer=initializer)(UpSampling2D(size=(2,2))(x11))
        x11 = concatenate([skip2, x11], axis=3)
        x11 = Conv2D(filters[7], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
        x11 = Conv2D(filters[7], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
        x11 = Dropout(drp_out)(x11)
        out = att(x11)
        out = att(out)
        skip8=out
        
        # nineth block
        x1 = layers.LayerNormalization(epsilon=1e-5)(out)
        x11=x1
        x11 = Conv2D(filters[8], 2, padding="same", activation=act, kernel_initializer=initializer)(UpSampling2D(size=(2,2))(x11))
        x11 = concatenate([skip1, x11], axis=3)
        x11 = Conv2D(filters[8], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
        x11 = Conv2D(filters[8], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
        x11 = Dropout(drp_out)(x11)
        out = att(x11)
        out = att(out)
        skip9=out

        skip7 = layers.LayerNormalization(epsilon=1e-5)(UpSampling2D(size=(2,2))(skip7))
        out7 = Conv2D(filters[6], 3, padding="same", activation=act, kernel_initializer=initializer)(skip7)
        out7 = Conv2D(filters[6], 3, padding="same", activation=act, kernel_initializer=initializer)(out7)
        #
        skip8 = layers.LayerNormalization(epsilon=1e-5)(UpSampling2D(size=(2,2))(skip8))
        out8 = Conv2D(filters[7], 3, padding="same", activation=act, kernel_initializer=initializer)(skip8)
        out8 = Conv2D(filters[7], 3, padding="same", activation=act, kernel_initializer=initializer)(out8)
        #
        skip9 = layers.LayerNormalization(epsilon=1e-5)(UpSampling2D(size=(2,2))(skip9))
        out9 = Conv2D(filters[8], 3, padding="same", activation=act, kernel_initializer=initializer)(skip9)
        out9 = Conv2D(filters[8], 3, padding="same", activation=act, kernel_initializer=initializer)(out9)
        #

        # # ACDC
        out7 = Conv2D(4, (1,1), activation="sigmoid", name='pred1')(out7)
        out8 = Conv2D(4, (1,1), activation="sigmoid", name='pred2')(out8)
        out9 = Conv2D(4, (1,1), activation="sigmoid", name='final')(out9)

        
        print("\n")
        print("DS 1 -> input:", skip7.shape, "output:", out7.shape) 
        print("DS 2 -> input:", skip8.shape, "output:", out8.shape) 
        print("DS 3 -> input:", skip9.shape, "output:", out9.shape) 
        


        model = keras.Model(inputs=inputs, outputs=[out7, out8, out9])

        
        return model    
    return create_model()