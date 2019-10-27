
####### IMPORTS #######
import os, errno, io, sys, glob, shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image
from tqdm import tqdm
from google_images_download import google_images_download

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.metrics import classification_report




def image_collector(topics,
                           img_count,
                           img_size):
    
    
    ####### DATA COLLECTION #######
    
    # creates a data directory if one does not exist. we will store images here
    try:
        os.makedirs('data')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
    # creates a 'train' and 'test' folders inside 'data' 
    # if there are some already, it creates another different ones with a number
    get_ipython().system('cd data')
    count = 1
    try:
        os.makedirs('data/train1')
    except:    
        if os.path.exists('data/train1') == True:
            while True:
                try:
                    os.makedirs('data/train' + str(count))
                    break
                except:
                    count+= 1
                    
                    
    # creating 2 lists in case topics contain 2 words
    # topics_search keeps both words, to use for googleimagesdownload, if applicable
    topics_clean = []
    topics_search = []
    for topic in topics:
        try:
            topic = topic.replace(' ', '_')
            topics_clean.append(topic.split('_')[0])
            topics_search.append(topic)
        except:
            topics_clean.append(topic)
    
    # creating new directory for each topic and collecting images for it
    print('Collecting images...')
    for topic_clean in topics_clean:
        try:
            os.makedirs('data/train' + str(count) + '/' + topic_clean)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        # getting the index from topics_clean, so that we can match it with topics, and use that instead to collect images
        topic_index = topics_clean.index(topic_clean)
        search_term = topics_search[topic_index]
        folder_name = topic_clean
        
        
        # image collection using google_images_download
        response = google_images_download.googleimagesdownload() 
        arguments = {'keywords': search_term,
                     'size': 'medium',
                     'limit': img_count, 
                     'format': 'jpg',
                     'time_range': '{"time_min":"01/01/2018","time_max":"12/01/2018"}',
                     'output_directory':'data/',
                     'image_directory': 'train' + str(count) + '/' + folder_name + '/',
                     'silent_mode': True,
                     'chromedriver': 'chromedriver.exe'}
        paths = response.download(arguments)
        
            
    ####### IMAGE PROCESSING #######
    
    X = []
    y = []
    
    print('Processing images...')
    for topic_clean in tqdm(topics_clean):
        # opening images in color, resizing them, and making each one into an array
        for f in glob.glob(os.path.join('data/train' + str(count), topic_clean, '*.jpg')):
            try:
                img = Image.open(str(f))
                img = img.convert('RGB')
                img = img.resize((img_size[0], img_size[1]))
                arr = image.img_to_array(img)
                
                # cropping images
                arr2d = extract_patches_2d(arr, patch_size= img_size)
                for crop in arr2d:
                    X.append(crop)
                    y.append(topic_clean)
                
            except:
                pass
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y




def model_maker(X, y, topics, epochs):
    # use GPU if available, to improve performance and fitting time
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    # vram limit for eficiency
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # changing format to satisfy keras
    X = preprocess_input(X)
    
    # preprocessing labels with scikit-learn LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    y_int = label_encoder.transform(y)
    
    # calculating class weight
    cls_weight = class_weight.compute_class_weight('balanced', np.unique(y_int), y_int)
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_int, test_size=0.2)
    
    # define image data generator for on-the-fly augmentation
    generator = image.ImageDataGenerator(zca_whitening=False, rotation_range=10,
                                     width_shift_range=0.1, height_shift_range=0.1,
                                     shear_range=0.02, zoom_range=0.1,
                                     channel_shift_range=0.05, horizontal_flip=True)
    
    # hides output, to avoid cluttering
    original_stdout = sys.stdout
    text_trap = io.StringIO()
    sys.stdout = text_trap
    
    # fit the generator (required if zca_whitening=True)
    generator.fit(X)
    
    # load the model without output layer for fine-tuning
    model_baseline = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    features = model_baseline.output
    # add output layer
    predictions = Dense(len(topics), activation='softmax')(features)

    # define the model
    model = Model(inputs=model_baseline.input, outputs=predictions)

    # freeze the resnet layers except the last 2 blocks
    for layer in model_baseline.layers[:154]:
        layer.trainable = False

    # compiling
    opt = Adam(lr=0.000005)
    opt_sgd = SGD(lr=5*1e-4, momentum=0.9)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
    model.summary()
    
    batch_size = 32
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    
    # restores terminal output
    sys.stdout = original_stdout
    
    # fitting model
    hist = model.fit_generator(generator.flow(X_train, y_train, batch_size=batch_size, shuffle=True),
                              epochs=epochs,
                              steps_per_epoch=int(X.shape[0])/batch_size,
                              class_weight=cls_weight,
                              validation_data=(X_test, y_test),
                              callbacks=[early_stop])
    
    
    # saving model. including classes and val score
    topics_clean = []
    topics_search = []
    for topic in topics:
        try:
            topic = topic.replace(' ', '_')
            topics_clean.append(topic.split('_')[0])
            topics_search.append(topic)
        except:
            topics_clean.append(topic)
    topics_str = ''
    for topic_clean in topics_clean:
        topics_str += topic_clean + '_'    
    model.save('model_' + topics_str + str(round(hist.history['val_acc'][-1], 2)) + '.h5')
    

    # plot the training loss and accuracy
    N = np.arange(0, epochs)
    plt.style.use("ggplot")
    plt.figure(figsize = (15, 8))
    plt.plot(N, hist.history["loss"], label="train loss", color = 'navy')
    plt.plot(N, hist.history["val_loss"], label="test loss", color = 'skyblue')
    plt.plot(N, hist.history["acc"], label="train accuracy", color = 'firebrick')
    plt.plot(N, hist.history["val_acc"], label="test accuracy", color = 'lightcoral')
    plt.title("Accuracy and Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    
    
    return model



def model_tester(model, topics, img_size):
    
    # deletes validation folder, in case this function has been ran already
    try:
        shutil.rmtree('data/validation')
    except:
        pass
    
    # creates a validation folder
    os.makedirs('data/validation')
    
    topics_clean = []
    topics_search = []
    for topic in topics:
        try:
            topic = topic.replace(' ', '_')
            topics_clean.append(topic.split('_')[0])
            topics_search.append(topic)
        except:
            topics_clean.append(topic)
    
    # collecting 1 image for each class
    for topic_clean in topics_clean:
        
        try:
            os.makedirs('data/validation/' + topic_clean)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        # getting the index from topics_clean, so that we can match it with topics, and use that instead to collect images
        topic_index = topics_clean.index(topic_clean)
        search_term = topics_search[topic_index]
        folder_name = topic_clean
        
        # hides output from terminal. google_images_download will clutter otherwise
        original_stdout = sys.stdout
        text_trap = io.StringIO()
        sys.stdout = text_trap
        
        # image collection using google_images_download
        response = google_images_download.googleimagesdownload() 
        arguments = {'keywords': search_term,
                     'size': 'medium',
                     'limit': 3, 
                     'format': 'jpg',
                     'time_range': '{"time_min":"01/01/2014","time_max":"12/01/2014"}',
                     'output_directory':'data/',
                     'image_directory': 'validation/' + folder_name + '/',
                     'silent_mode': True,
                     'chromedriver': 'chromedriver.exe'}
        paths = response.download(arguments)

        # restores terminal output
        sys.stdout = original_stdout
    
    # for each image, show the image and use model to predict % chance for each class
    for topic_clean in topics_clean:
    
        topic_index = topics_clean.index(topic_clean)
        print('Image #'+ str(topic_index + 1) +' from ' + topic_clean + ' class: \n')
        
        list_of_img = glob.glob('data/validation/' + topic_clean + '/*')
        sorted_files = sorted(list_of_img, key=os.path.getmtime)
        
        path = sorted_files[0].replace('\\', '/')
        
        img = Image.open(path)
        img = img.convert('RGB')
        img = img.resize((img_size[0], img_size[1]))
        
        # making prediction on img
        img_pred = np.expand_dims(img, axis=0)
        pred = model.predict(img_pred)
        
        
        # setting up to show multiple images
        rows = len(topics)
        fig = plt.figure(figsize = (25, 25))
        fig.add_subplot(rows, 1, topic_index + 1)
        plt.imshow(img)
        
        #print probability of each image being each class
        for sub_topic in topics:
            subtopic_index = topics.index(sub_topic)
            print(f' The model predicted there is a {round((pred[0][subtopic_index]) * 100, 2)} % chance this is a {sub_topic}') 
        print('------------------------------------\t')
    
    return

