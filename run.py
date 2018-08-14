#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.


from mvnc import mvncapi as mvnc
import sys
import numpy
import cv2
import time
import csv
import os
import sys
import re
import time

dim=(224,224)
EXAMPLES_BASE_DIR='../../'
imagename = 'cat.jpg'

def infer(imgname):

        # get labels
        #labels_file=EXAMPLES_BASE_DIR+'data/ilsvrc12/synset_words.txt'
        #labels=numpy.loadtxt(labels_file,str,delimiter='\t')
        
        
        # Get a list of ALL the sticks that are plugged in
        devices = mvnc.enumerate_devices()
        if len(devices) == 0:
                print('No devices found')
                quit()
        
        # Pick the first stick to run the network
        device = mvnc.Device(devices[0])
        
        # Open the NCS
        device.open()

        # set the file name of the compiled network (graph file)
        #file_dir = os.path.dirname(os.path.realpath(__file__))
        network_filename = "graph"#file_dir + '/graph'

        # Load network graph file into memory
        with open(network_filename, mode='rb') as net_file:
                memory_graph = net_file.read()

        # create and allocate the graph object
        graph = mvnc.Graph("Mobilenet_v2")
        fifo_in, fifo_out = graph.allocate_with_fifos(device, memory_graph)

        # Load the image and preprocess it for the network
        #ilsvrc_mean = numpy.load(EXAMPLES_BASE_DIR+'data/ilsvrc12/ilsvrc_2012_mean.npy').mean(1).mean(1) #loading the mean file
        cap = cv2.VideoCapture(0)
        while(1):
            # read the image to run an inference on from the disk
            T1=time.time()
            cap.grab()
            ret,frame=cap.retrieve()
            T2=time.time()
            #infer_image = cv2.imread(IMAGE_FULL_PATH)
            infer_image=frame
            # run a single inference on the image and overwrite the
            # boxes and labels
        


            img = infer_image
            h, w, _ = img.shape
            if h < w:
                off = int((w - h) / 2)
                img= img[:, off:off + h]
            else:
                off = int((h - w) / 2)
                img= img[off:off + h, :]
            img=cv2.resize(img,dim)
            img = img.astype(numpy.float32)
            img[:,:,0] = (img[:,:,0] - 103.94)
            img[:,:,1] = (img[:,:,1] - 116.78)
            img[:,:,2] = (img[:,:,2] - 123.68)
            img = img * 0.017
            T3=time.time()
            # Send the image to the NCS and queue an inference
            graph.queue_inference_with_fifo_elem(fifo_in, fifo_out, img.astype(numpy.float32), None)

            # Get the result from the NCS by reading the output fifo queue
            output, userobj = fifo_out.read_elem()
            T4=time.time()    
            OUT_NCS = numpy.squeeze(output)
            #print(OUT_NCS)
            idx = numpy.argsort(-OUT_NCS)
            label_names = numpy.loadtxt('synset.txt', str, delimiter='\t')
            for i in range(5):
                label = idx[i]
                #print('%.2f - %s' % (OUT_NCS[label], label_names[label]))
                cv2.putText(infer_image, label_names[label], (20, 20+10*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
            # display the results and wait for user to hit a key
            cv2.imshow("MobileNet_V2", infer_image)
            cv2.waitKey(1)
            T5 = time.time()
            print("Total Time:"+str(T5-T1)+"  getFrame:"+str(T2-T1)+"  preProcess:"+str(T3-T2)+"  inferenfe:"+str(T4-T3)+"  Display"+str(T5-T4))
        
        # Clean up the graph, device, and fifos
        fifo_in.destroy()
        fifo_out.destroy()
        graph.destroy()
        device.close()
        device.destroy()
        return

if __name__ == "__main__":
        infer(imagename)
        #print (result)




