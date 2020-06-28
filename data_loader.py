import numpy as np
from PIL import Image
from scipy.optimize import curve_fit
import random

def fit_func(data,a,b,c,d,e,f):
    l_x = len(data[:,0])
    l_y = len(data[0])

    y = data[0]
    x = np.repeat(range(1,l_x+1), l_y)
    y = np.tile(range(1,len(data[0])+1), ((l_x,1))).flatten()

    '''print("X: ")
    print(x)
    print(x.shape)
    print("Y: ")
    print(y)
    print(y.shape)'''

    f = a+b*x+c*y+d*x*y+e*np.power(x,2)+f*np.power(y,2)
    return f

def curve_params(pp):
    x,y = pp.shape
    lin_x = np.linspace(1,x,x)
    lin_y = np.linspace(1,y,y)
    tile_x = np.tile(lin_y,(x,1))
    tile_y = np.tile(lin_x.transpose(),(y,1)).transpose()

    xy = np.tile(lin_x.flatten(),(y,1))
    z = pp.transpose().flatten()

    popt, pcov = curve_fit(fit_func,xy,z) # , bounds=([0,0,0,0,0,0],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])

    '''print(tile_x)
    print(tile_y)

    print(tile_x.shape)
    print(tile_y.shape)
    print((tile_x*tile_y).shape)
    print(np.power(tile_y,2).shape)

    print("Popt")
    print(popt)
    print(pcov)

    print("PP shape")
    print(pp.shape)

    print("Z: ")
    print(z)'''

    return popt

def curve_removal(pp, params=None):
    x,y = pp.shape
    lin_x = np.linspace(1,x,x)
    lin_y = np.linspace(1,y,y)
    tile_x = np.tile(lin_y,(x,1))
    tile_y = np.tile(lin_x.transpose(),(y,1)).transpose()

    if params is not None:
        popt = params
    else:
        popt = curve_params(pp)

    return popt[0]+ popt[1]*tile_x+popt[2]*tile_y+popt[3]*tile_x*tile_y+popt[4]*np.power(tile_x,2)+popt[5]*np.power(tile_y,2)

def covar(a,b):
    ma = np.mean(a)
    mb = np.mean(b)
    am, bm = a-ma, b-mb
    s = np.sum(np.multiply(am,bm))
    
    #return s/tf.cast((tf.size(a)[-1]-1),tf.float32)
    return s

def npcc(y_true, y_pred):
    r_num = covar(y_true,y_pred)
    r_den = np.sqrt(np.multiply(covar(y_true,y_true),covar(y_pred,y_pred)))
    r = r_num / r_den

    r = np.maximum(np.minimum(r, 1.0), -1.0)
    return -r

def mse(y_true,y_pred):
    return np.mean(np.power(y_true-y_pred,2))

class data_loader:
    def __init__(self,max_x,max_y):
        self.max_x = max_x
        self.max_y = max_y
    
    def get_data_large(self, folder,noisy_files,clean,duplicates,threshold=0.25,noisy=False,flipped=False, augment=False):    
        x_list = []
        y_list = []

        files = []

        noisy_imgs = []
        clean_imgs = []

        filelist = []

        with Image.open(f'{folder}{clean[0]}') as c_img:
            #if len(c_img.shape) == 3:
            #    c_img = c_img.convert('L')
            clean_img = np.asarray(c_img)

        if flipped:
            clean_img = np.flip(clean_img,axis=0)

        curve = curve_removal(clean_img,None)
        clean_min = np.min(clean_img-curve)
        curveless_clean = clean_img-curve-clean_min

        #curveless_clean = (curveless_clean-np.mean(curveless_clean))/np.std(curveless_clean)

        for a in range(len(noisy_files)):
            with Image.open(f'{folder}{noisy_files[a]}') as n_img:
                noisy_img = np.asarray(n_img) #.convert('L')

                if flipped:
                    noisy_img = np.flip(noisy_img,axis=0)

                curve = curve_removal(noisy_img,None)
                noisy_min = np.min(noisy_img-curve)
                curveless_noisy = noisy_img-curve-noisy_min

                #curveless_noisy = (curveless_noisy-np.mean(curveless_noisy))/np.std(curveless_noisy)

                noisy_imgs.append(curveless_noisy)

        for a in range(len(noisy_files)): 
            for i in range(duplicates):
                #clean_img = np.asarray(Image.open(f'{folder}{clean[0]}.tif'))
                noisy_img = noisy_imgs[a] #np.asarray(Image.open(f'{folder}{noisy[a]}.tif'))
                noisy_npcc = 0
                c = 0 

                while noisy_npcc < threshold and c < 1000:
                    #x_list.append(np.load(f'{self.noisy_folder}{file}').reshape((max_x,max_y,1)))
                    shifted_x = np.random.randint(clean_img.shape[0]-self.max_x)
                    shifted_y = np.random.randint(clean_img.shape[1]-self.max_y)

                    cropped_noisy = noisy_img[shifted_x:shifted_x+self.max_x,shifted_y:shifted_y+self.max_y]

                    if noisy:
                        noise = np.random.normal(scale=1,size=(self.max_x,self.max_y))
                        #noise = sp.misc.imresize(noise,(max_x,max_y))
                        cropped_noisy = cropped_noisy + noise

                    cropped_clean = curveless_clean[shifted_x:shifted_x+self.max_x,shifted_y:shifted_y+self.max_y]

                    noisy_npcc = abs(npcc(cropped_noisy,cropped_clean))
                    c += 1

                #curve = curve_removal(cropped_noisy,None)
                #noisy_min = np.min(cropped_noisy-curve)
                #curveless_noisy = cropped_noisy-curve-noisy_min

                cropped_noisy = (cropped_noisy-np.mean(cropped_noisy))/(np.std(cropped_noisy))
                cropped_clean = (cropped_clean-np.mean(cropped_clean))/(np.std(cropped_clean))

                if noisy_npcc >= threshold:
                    
                    
                #curve = curve_removal(cropped_clean,None)
                #clean_min = np.min(cropped_clean-curve)
                #curveless_clean = cropped_clean-curve-clean_min
                    if augment:
                        if np.random.randint(10) < 2:
                            cropped_noisy = np.flip(cropped_noisy,axis=0)
                            cropped_clean = np.flip(cropped_clean,axis=0)
                        
                        if np.random.randint(10) < 2:
                            cropped_noisy = np.flip(cropped_noisy,axis=1)
                            cropped_clean = np.flip(cropped_clean,axis=1)
                
                    #y_img_fft = np.fft.fftshift(np.fft.fft2(cropped_clean))
                    #y_img_bpass = y_img_fft.copy()
                    #y_img_bpass[200:300,200:300] = 0
                    #y_img_bpass_ifft = np.abs(np.fft.ifft2(np.fft.ifftshift(y_img_bpass)))
                
                    x_list.append(cropped_noisy.reshape((self.max_x,self.max_y,1)))
                    y_list.append(cropped_clean.reshape((self.max_x,self.max_y,1)))
                    #y_list.append(y_img_bpass_ifft.reshape((self.max_x,self.max_y,1)))
                    filelist.append(noisy_files[a])
                else:
                    print("Failed to find suitable image:",noisy_files[a])
                    break

        x_array = np.asarray(x_list)
        y_array = np.asarray(y_list)
                        
        return x_array, y_array, filelist
    
    def get_bead_ds(self):
        folder = "data/observed/beads_40x_0_70NA/6um/02_09_2019/"

        train_data_x = None
        train_data_y = None

        train_input_filelist_2 = ['NE04NE20_NE10_2','NE02NE20_NE10_2','NE20_NE10_2'] #
        train_target_filelist_2 = ['NE00_NE10_2']

        train_input_filelist_3 = ['NE04NE20_NE10_f2','NE02NE20_NE10_f2','NE20_NE10_f2'] #
        train_target_filelist_3 = ['NE00_NE10_f2']

        test_input_filelist = ['NE01NE20+NE10','NE03NE20+NE10','NE04NE20+NE10','NE20+NE10','NE02NE20+NE10']
        test_target_filelist = ['NE00+NE10']

        matched_files = [4,2,5,6,7,12,13,14,15,16,19]
        f = 'data/observed/beads_40x_0_70NA/6um/matched_fringes/17_09_2019/'
        test_data_x, test_data_y, test_filelist = self.get_data_large(f,[f'NE06NE20_{matched_files[0]}.tif'],[f'NE00_{matched_files[0]}.tif'],60,0.2,False,flipped=True)

        for i in range(1,len(matched_files)): # 
            data_x,data_y,data_filelist = self.get_data_large(f,[f'NE06NE20_{matched_files[i]}.tif'],[f'NE00_{matched_files[i]}.tif'],30,0.2,False,flipped=True)

            if train_data_x is None:
                train_data_x = data_x
                train_data_y = data_y
                train_filelist = data_filelist
            else:
                train_data_x = np.append(train_data_x,data_x,axis=0)
                train_data_y = np.append(train_data_y,data_y,axis=0)
                train_filelist = np.append(train_filelist,data_filelist,axis=0)

        matched_files = [0,2,7,8,9,12,13,14,15,18,20]
        f = 'data/observed/beads_40x_0_70NA/6um/matched_fringes/'

        for i in range(1,len(matched_files)): # 
            data_x,data_y,data_filelist = self.get_data_large(f,[f'NE03NE20_{matched_files[i]}.tif'],[f'NE00_{matched_files[i]}.tif'],30,0.2,False,flipped=True)

            if train_data_x is None:
                train_data_x = data_x
                train_data_y = data_y
                train_filelist = data_filelist
            else:
                train_data_x = np.append(train_data_x,data_x,axis=0)
                train_data_y = np.append(train_data_y,data_y,axis=0)
                train_filelist = np.append(train_filelist,data_filelist,axis=0)

        matched_files = [1,7,9,11,12,16,18,19,20]
        f = 'data/observed/beads_40x_0_70NA/6um/matched_fringes/15_09_2019/'
        test_data_x_2, test_data_y_2, test_filelist_2 = self.get_data_large(f,[f'NE10NE20_{matched_files[0]}.tif'],[f'NE00_{matched_files[0]}.tif'],60,0.2,False,flipped=True)

        test_data_x = np.append(test_data_x,test_data_x_2,axis=0)
        test_data_y = np.append(test_data_y,test_data_y_2,axis=0)
        test_filelist = np.append(test_filelist,test_filelist_2,axis=0)

        for i in range(1,len(matched_files)): # 
            data_x,data_y,data_filelist = self.get_data_large(f,[f'NE10NE20_{matched_files[i]}.tif'],[f'NE00_{matched_files[i]}.tif'],30,0.2,False,flipped=True)

            if train_data_x is None:
                train_data_x = data_x
                train_data_y = data_y
                train_filelist = data_filelist
            else:
                train_data_x = np.append(train_data_x,data_x,axis=0)
                train_data_y = np.append(train_data_y,data_y,axis=0)
                train_filelist = np.append(train_filelist,data_filelist,axis=0)

        return train_data_x, train_data_y, test_data_x, test_data_y, train_filelist, test_filelist

    def get_bj7_ds(self, num_imgs=100, threshold=0.1):
        folder = "data/observed/20x_0_45NA/19_11_19/"

        #train_input_filelist = ['NE20_1'] #'NE10NE20_2','NE01NE10NE20_2','NE02NE10NE20_2','NE03NE10NE20_2','NE04NE10NE20_2'
        #train_target_filelist = ['NE00_1']

        #test_input_filelist = ['NE20_5'] #,'NE10NE20_5','NE01NE10NE20_5','NE02NE10NE20_5','NE03NE10NE20_5','NE04NE10NE20_5'
        #test_target_filelist = ['NE00_5']

        train_input_filelist = []
        train_target_filelist = []

        test_input_filelist = []
        test_target_filelist = []

        train_data_x = None
        train_data_y = None

        test_data_x = None
        test_data_y = None

        #train_list = [20,1,2,4,6,7,8,9,11,13]
        #test_list = [18,16,15,14]
        
        train_list = [1,2,5,6,8,9,10]
        test_list = [14,15,18,20]
        
        
        for i in train_list:
            train_input_filelist.append(f'NE24_{i}')
            train_target_filelist.append(f'NE00_{i}')

            img_data_x,img_data_y, data_filelist = self.get_data_large(folder,[f'NE24_{i}.tif'],[f'NE00_{i}.tif'],100,0.02,False,augment=True)

            if len(img_data_x.shape) != 4:
                continue

            if train_data_x is None:
                train_data_x = img_data_x.copy()
                train_data_y = img_data_y.copy()
            else:
                train_data_x = np.append(train_data_x,img_data_x,axis=0)
                train_data_y = np.append(train_data_y,img_data_y,axis=0)

        for i in test_list:
            test_input_filelist.append(f'NE24_{i}')
            test_target_filelist.append(f'NE00_{i}')

            img_data_x, img_data_y, data_filelist = self.get_data_large(folder,[f'NE24_{i}.tif'],[f'NE00_{i}.tif'],100,0.02,False)

            if len(img_data_x.shape) != 4:
                continue

            if test_data_x is None:
                test_data_x = img_data_x.copy()
                test_data_y = img_data_y.copy()
            else:
                test_data_x = np.append(test_data_x,img_data_x,axis=0)
                test_data_y = np.append(test_data_y,img_data_y,axis=0)

        #train_data_x,train_data_y = get_data_large(['NE_30A_4','NE_30A_6'],['NE_20A_4','NE_30A_6'],800,False)
        #test_data_x, test_data_y = get_data_large(['NE_30A_5'],['NE_20A_5'],80,True)    

        return train_data_x, train_data_y, test_data_x, test_data_y, train_input_filelist, test_input_filelist
    
    def get_bj7_ds_live(self, num_imgs=100, threshold=0.1):
        folder = "data/observed/20x_0_45NA/2_12_19/"
        
        train_input_filelist = []
        train_target_filelist = []

        test_input_filelist = []
        test_target_filelist = []

        train_data_x = None
        train_data_y = None

        test_data_x = None
        test_data_y = None
        
        for i in range(1,15):
            train_input_filelist.append(f'NE30_{i}')
            train_target_filelist.append(f'NE00_{i}')

            img_data_x,img_data_y, data_filelist = self.get_data_large(folder,[f'NE30_{i}.tiff'],[f'NE00_{i}.tif'],100,0.02,False,augment=True)

            if len(img_data_x.shape) != 4:
                continue

            if train_data_x is None:
                train_data_x = img_data_x.copy()
                train_data_y = img_data_y.copy()
            else:
                train_data_x = np.append(train_data_x,img_data_x,axis=0)
                train_data_y = np.append(train_data_y,img_data_y,axis=0)

        for i in range(15,21):
            test_input_filelist.append(f'NE30_{i}')
            test_target_filelist.append(f'NE00_{i}')

            img_data_x, img_data_y, data_filelist = self.get_data_large(folder,[f'NE30_{i}.tiff'],[f'NE00_{i}.tif'],num_imgs,0.02,False)

            if len(img_data_x.shape) != 4:
                continue

            if test_data_x is None:
                test_data_x = img_data_x.copy()
                test_data_y = img_data_y.copy()
            else:
                test_data_x = np.append(test_data_x,img_data_x,axis=0)
                test_data_y = np.append(test_data_y,img_data_y,axis=0)

        #train_data_x,train_data_y = get_data_large(['NE_30A_4','NE_30A_6'],['NE_20A_4','NE_30A_6'],800,False)
        #test_data_x, test_data_y = get_data_large(['NE_30A_5'],['NE_20A_5'],80,True)    

        return train_data_x, train_data_y, test_data_x, test_data_y, train_input_filelist, test_input_filelist