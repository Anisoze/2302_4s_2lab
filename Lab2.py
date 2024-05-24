

def ask(num1,num2):         #ask option
    while(True):
        f=(input())
        if(len(f)==0 or ord(f)<ord(str(num1)) or ord(f)>ord(str(num2))):
            print("\nwrong input\n")
        else:
            return int(f)
        






def open_image():           #open image
    print("\nchoose Image:\n1 - image_552.jpg\n2 - image80.jpg\n3 - image_s.jpg\n4 - image_ss.jpg\n5 - png_image.png\n6 - png_img\n7 - from RAW\n8 - other\n") #choose image
    f2 = ask(1,8)
        
    if (f2==1):
        filename="image_552.jpg"
        with Image.open(filename) as img:
            img.load()
            

    elif(f2==2):
        filename="image80.jpg"
        with Image.open(filename) as img:
            img.load()
    #elif(f2==2):
        #filename="YCbCr_image"
        #file=open(filename,"rb")
        #A=np.load(file)
        #file.close()
        #img=Image.fromarray(A, mode="YCbCr")
            
    elif(f2==3):
        filename="image_s.jpg"
        with Image.open(filename) as img:
            img.load()
            
    elif(f2==4):
        filename="image_ss.jpg"
        with Image.open(filename) as img:
            img.load()
            
    elif(f2==5):
        filename="png_image.png"
        with Image.open(filename) as img:
            img.load()


    elif(f2==6):
        file=open("png_img","rb")
        A=np.load(file)
        file.close()

        img=Image.fromarray(A, mode="RGB")
        return img




    elif(f2==7):        #from RAW
        img=RAW(0)
        


    elif(f2==8):                    #other
        print("\ninput name\n")
        filename = input()
        print("\nchoose type:\n1 - RGB\n2 - YCbCr\n")
        im=ask(1,2)
        if(im==1):
            with Image.open(filename) as img:
                img.load()
        else:
            file=open(filename,"rb")
            A=np.load(file)
            file.close()
            img=Image.fromarray(A, mode="YCbCr")
    return img







def RGB_to_YCbCr(img):              #turn RGB to YCbCr
    #mode = 1 # return img
    mode = 2 # return arr
    
    #R=np.zeros((img.height, img.width))
    #G=np.zeros((img.height, img.width))
    #B=np.zeros((img.height, img.width))
    img_arr=np.asarray(img)
    #for j in range(img.height):
        #for i in range(img.width):
            #R[j][i]=img_arr[j][i][0]
            #G[j][i]=img_arr[j][i][1]
            #B[j][i]=img_arr[j][i][2]
    R=np.asarray([ [ img_arr[j][i][0] for i in range(img.width)] for j in range(img.height)])
    G=np.asarray([ [ img_arr[j][i][1] for i in range(img.width)] for j in range(img.height)])
    B=np.asarray([ [ img_arr[j][i][2] for i in range(img.width)] for j in range(img.height)])
                

    Y=np.asarray([ [ R[j][i]/1000*299 + G[j][i]/1000*587 + B[j][i]/1000*114 for i in range(img.width)] for j in range(img.height)])
    Cb=np.asarray([ [ R[j][i]/1000000*(-168736) + G[j][i]/1000000*(-331264) + B[j][i]/2 + 128 for i in range(img.width)] for j in range(img.height)])
    Cr=np.asarray([ [ R[j][i]/2 + G[j][i]/1000000*(-418688) + B[j][i]/1000000*(-81312) + 128 for i in range(img.width)] for j in range(img.height)])
          
    img_arr=np.asarray([ [ [Y[j][i], Cb[j][i], Cr[j][i]] for i in range(img.width)] for j in range(img.height)])
    
    if(mode==1):        #return img
        
        #Y_im=Image.fromarray(Y.astype("uint8")).convert('L')               
        #Cb_im=Image.fromarray(Cb.astype("uint8")).convert('L')
        #Cr_im=Image.fromarray(Cr.astype("uint8")).convert('L')
                   
        Y_im=Image.fromarray(img_arr.astype("uint8"), "YCbCr")
        #Y_im.show()    
        #Cb_im.show()  
        #Cr_im.show()  

        #Y_im.show()  
        return Y_im 
    
    else:       #return array
        return img_arr




def YCbCr_to_RGB(img):          #turn YCbCr to RGB
    
    #num_rows = img.height
    #num_columns = img.width
    #data = img.load()
    #arr = np.zeros((num_rows, num_columns, 3), dtype=np.uint8)
    #Y=[ []for i in range(len(arr)) ]
    #arr2=[[]for i in range(len(arr))]
    #arr2=[[]for i in range(len(arr))]
    #for y in range(num_rows):
        #for x in range(num_columns):
            #for i in range(n):
                #arr[y, x, i] = data[x, y][i]
                #Y[i].append(arr[i][j][ch])   
                #Cb[i].append(arr[i][j][ch])   
                #Cr[i].append(arr[i][j][ch])   
    
    

    img_arr=np.asarray(img)
    Y=np.asarray([ [ img_arr[j][i][0] for i in range(img.width)] for j in range(img.height)])
    Cb=np.asarray([ [ img_arr[j][i][1] for i in range(img.width)] for j in range(img.height)])
    Cr=np.asarray([ [ img_arr[j][i][2] for i in range(img.width)] for j in range(img.height)])
                


    R=np.asarray([ [ (Y[j][i]) + (Cb[j][i]-128)/922967728*(-1125) + (Cr[j][i]-128)/922967728*10352003*125 for i in range(img.width)] for j in range(img.height)])
    G=np.asarray([ [ Y[j][i] + (Cb[j][i]-128)/922967728*(-317626125) + (Cr[j][i]-128)/922967728*(-659124625) for i in range(img.width)] for j in range(img.height)])
    B=np.asarray([ [ Y[j][i] + (Cb[j][i]-128)/922967728*13083991*125 + (Cr[j][i]-128)/922967728*375 for i in range(img.width)] for j in range(img.height)])

                
    #R_im=Image.fromarray(R.astype("uint8")).convert('L')               
    #G_im=Image.fromarray(G.astype("uint8")).convert('L')
    #B_im=Image.fromarray(B.astype("uint8")).convert('L')
                
    #R_im.show()    
    #G_im.show()  
    #B_im.show()  

    d_img_arr=np.asarray([ [ [R[j][i], G[j][i], B[j][i]] for i in range(img.width)] for j in range(img.height)])
                
    RGB_im=Image.fromarray(d_img_arr.astype("uint8"), "RGB")
    #RGB_im.show()   
    return RGB_im




def make_arr(img,n):            #make image to array
    num_rows = img.height
    num_columns = img.width
    data = img.load()
    arr = np.zeros((num_rows, num_columns, n), dtype=np.uint8)
    for y in range(num_rows):
        for x in range(num_columns):
            for i in range(n):
                arr[y, x, i] = data[x, y][i]
    return arr







def pick_channel(arr, ch):                  #pick one channel of array
    arr2=np.asarray([ [ arr[j][i][ch] for i in range(img.width)] for j in range(img.height)])
    #arr2=[[]for i in range(len(arr))]
    #for i in range(len(arr)):
        #for j in range(len(arr[0])):
            #arr2[i].append(arr[i][j][ch])    
    return arr2







def matrix_traversal(arr):              #matrix_traversal
    arr2 = []
    rows = len(arr)
    col = len(arr[0])

    x = 0
    y = -1
    all_diagonals = rows + col - 1
    for diagonal in range(all_diagonals):
        if diagonal % 2 == 0:
            y += 1
            while x > -1 and y < rows:
                arr2.append(arr[y][x])
                x -= 1
                y += 1
            if y == rows:
                y -= 1
                x += 1
        else:
            x += 1
            while y > -1 and x < col:
                arr2.append(arr[y][x])
                y -= 1
                x += 1
            if x == col:
                x -= 1
                y += 1
    return arr2





def line_to_matrix(line, rows, col):
    arr=np.zeros((rows, col))
    x=0
    y=-1
    i=0
    all_diagonals = rows + col - 1
    for diagonal in range(all_diagonals):
        if(diagonal%2==0):
            y+=1
            while(x>-1 and y<rows):
                arr[y][x]=line[i]
                i+=1
                x-=1
                y+=1
            if(y==rows):
                y-=1
                x+=1
        else:
            x+=1
            while(y>-1 and x<col):
                arr[y][x]=line[i]
                i+=1
                y-=1
                x+=1
            if(x==col):
                x-=1
                y+=1
    return arr
                
        




    
def down_rows_and_columns(arr, modX, modY):                     #downsaple with rows and columns
    arr2=[[] for i in range(math.ceil(len(arr)/modX)) ]
    
    for i in range(math.ceil(len(arr)/modX)):
        for j in range(math.ceil(len(arr[0])/modY)):
            arr2[i].append(arr[i*modX][j*modY])
    
    #print("\n\n")
    #print(arr)
    #print("\n\n")
    #print(arr2)
    return arr2
    







def down_average(arr, modX, modY):                                  #downsaple with average value
    arr2=[ [] for i in range(math.ceil(len(arr)/modX)) ]
    
    for i in range(math.ceil(len(arr)/modX)):
        for j in range(math.ceil(len(arr[0])/modY)):
            value=0
            t=True
            for x in range(modX):
                for y in range(modY):
                    if ((i*modX+x < len(arr)) and (j*modY+y < len(arr[0])) ):                     
                        value+=arr[i*modX+x][j*modY+y]/(modX*modY) 
                    else:
                        t=False
            if(t):
                #arr2[i].append(math.ceil(value))
                arr2[i].append(value)

    #print("\n\n")
    #print(arr)
    #print("\n\n")
    #print(arr2)

    return arr2








def down_next_to_average(arr, modX, modY):                              #downsaple with next to average value
    arr2=[ [] for i in range(math.ceil(len(arr)/modX)) ]
    
    for i in range(math.ceil(len(arr)/modX)):
        for j in range(math.ceil(len(arr[0])/modY)):
            value=0
            t=True
            for x in range(modX):
                for y in range(modY):
                    if ((i*modX+x < len(arr)) and (j*modY+y < len(arr[0])) ):                     
                        value+=arr[i*modX+x][j*modY+y]/(modX*modY) 
                    else:
                        t=False
                        break
            if(t):
                min_dif=math.inf
                v2=0
                for x in range(modX):
                    for y in range(modY):
                        if ((i*modX+x < len(arr)) and (j*modY+y < len(arr[0])) ):                     
                            dif=abs(value-arr[i*modX+x][j*modY+y])
                            if(dif<min_dif):
                                min_dif=dif
                                v2=arr[i*modX+x][j*modY+y]
                        else:
                            t=False
                            break
                arr2[i].append(math.ceil(v2))
                #print(value)
                #print(v2)

    #print("\n\n")
    #print(arr)
    #print("\n\n")
    #print(arr2)
    return arr2







def upsample(arr, modX, modY):                     #upsample
    arr2=[ [] for i in range(len(arr)*modX) ]
    
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            for x in range(modX):
                for y in range(modY):
                    arr2[i*modX+x].append(arr[i][j])


    #print("\n\n")
    #print(arr)
    #print("\n\n")
    #print(arr2)
    return arr2












def RAW(img):               #work with RAW of image
    
    print("\nchoose action:\n1 - save raw\n2 - open raw\n")               
    f = ask(1,2)
                
    if (f==1):      #save raw
        print("\nchoose out:\n1 - saved_raw\n2 - other\n")        
        f2=ask(1,2)
        if(f2==1):
            name="saved_raw"
        else:
            print("\nenter name\n")
            name=input()
            
        print("\nchoose mode:\n1 - RGB\n2 - L\n")
        f2=ask(1,2)
        if(f2==1):
            rgb=True
        else:
            rgb=False
            
                        #save raw as array
        if(rgb):    #rgb
            obj=img.load()
            B=np.zeros([img.height,img.width, 3], dtype=np.uint8)
            #data2=bytearray()                
            for i in range(img.height):
                for j in range(img.width):
                    for c in range (3):
                        B[i,j, c]=obj[j, i][c]
                        #data2.append(obj[j, i][c])
            file=open(name,"wb")
            np.save(file, B)
            file.close()
                

        else:   #not rgb
            obj=img.load()
            B=np.zeros([img.height,img.width], dtype=np.uint8)
            for i in range(img.height):
                for j in range(img.width):
                    B[i,j]=obj[j, i]
            file=open(name,"wb")
            np.save(file, B)
            file.close()
                
        return img
                    
                    

    else:           #open raw
        
        while(True):
            print("\nchoose open:\n1 - saved_raw\n2 - other\n")
            f2=ask(1,2)
            if(f2==1):
                name="saved_raw"  
            
            elif(f2==2):
                print("\nenter name\n")
                name=input()
                    
            file=open(name,"rb")
            A=np.load(file)
            file.close()
                
            print("\nchoose mode:\n1 - RGB\n2 - L\n")
            f2=ask(1,2)
            if(f2==1):
                img=Image.fromarray(A, mode="RGB")
                #img=Image.frombytes('RGB', )
                return img
            else:
                img=Image.fromarray(A, mode="L")
                return img
         






def coef(v):                        #coef for DCT
    if(v==0):
        return 1/math.sqrt(2)
    else:
        return 1








def DCT(arr):                   #DCT
    arr2 = np.zeros((8, 8))  
    
    for i in range(8):
        for j in range(8):       
            arr2[j][i]=1/4 * coef(i) * coef(j) * np.sum( [ [arr[y][x] * math.cos((2*x+1)*i*math.pi/16) * math.cos((2*y+1)*j*math.pi/16) for y in range(8)] for x in range(8) ])


    
    #print("\n\n")
    #print(arr)
    #print("\n\n")
    #print(arr2)

    return arr2






def back_DCT(arr):                  #reverse DCT
    arr2 = np.zeros((8, 8))  
    
    for x in range(8):
        for y in range(8):       
            arr2[y][x]=1/4 * np.sum( [ [coef(j) * coef(i) * arr[j][i] * math.cos((2*x+1)*i*math.pi/16) * math.cos((2*y+1)*j*math.pi/16) for j in range(8) ] for i in range(8) ] )


    
    #print("\n\n")
    #print(arr)
    #print("\n\n")
    #print(arr2)

    return arr2







def to_DCT(arr, arr2, two):             #general DCT cycle

    if(two):                                                    #two arrays
        arr_h=[[] for i in range(math.ceil(len(arr)/8))]
        arr_h2=[[] for i in range(math.ceil(len(arr2)/8))]
        for i in range(len(arr_h)):
            for j in range(math.ceil(len(arr[0])/8)):
                arr_tmp=np.zeros((8, 8))  
                arr_tmp2=np.zeros((8, 8)) 
                for x in range(8):
                    for y in range(8):
                        if(8*i+x < len(arr) and 8*j+y < len(arr[0])):
                            arr_tmp[x][y]=arr[8*i+x][8*j+y]
                            arr_tmp2[x][y]=arr2[8*i+x][8*j+y]
                arr_tmp=arr_tmp-127
                arr_tmp2=arr_tmp2-127
                arr_h[i].append(DCT(arr_tmp))
                arr_h2[i].append(DCT(arr_tmp2))
                
        return arr_h, arr_h2
                
    else:
        arr_h=[[] for i in range(math.ceil(len(arr)/8))]        #one array
        for i in range(len(arr_h)):
            for j in range(math.ceil(len(arr[0])/8)):
                arr_tmp=np.zeros((8, 8))  
                for x in range(8):
                    for y in range(8):
                        if(8*i+x < len(arr) and 8*j+y < len(arr[0])):
                            arr_tmp[x][y]=arr[8*i+x][8*j+y]
                arr_tmp=arr_tmp-127
                arr_h[i].append(DCT(arr_tmp))

        return arr_h
        






def reverse_from_DCT(arr_h, arr_h2, two):         #general reverse DCT cycle

    if(two):
        arr_b=np.zeros((8*len(arr_h), 8*len(arr_h[0])))      #two arrays
        arr_b2=np.zeros((8*len(arr_h2), 8*len(arr_h2[0]))) 
        for i in range(len(arr_h)):
            for j in range(len(arr_h[0])):
                arr_tmp=back_DCT(arr_h[i][j])
                arr_tmp=arr_tmp+127
                arr_tmp2=back_DCT(arr_h2[i][j])
                arr_tmp2=arr_tmp2+127
                for x in range(8):
                    for y in range(8):
                        arr_b[i*8+x][j*8+y]=arr_tmp[x][y]
                        arr_b2[i*8+x][j*8+y]=arr_tmp2[x][y]
                        
        return arr_b, arr_b2

        
    else:
        arr_b=np.zeros((8*len(arr_h), 8*len(arr_h[0])))      #one array
        for i in range(len(arr_h)):
            for j in range(len(arr_h[0])):
                arr_tmp=back_DCT(arr_h[i][j])
                arr_tmp=arr_tmp+127
                for x in range(8):
                    for y in range(8):
                        arr_b[i*8+x][j*8+y]=arr_tmp[x][y]
                        
        return arr_b 
            












def mult_matrix(A, B):
    C=[[] for i in range(len(B))]
    for i in range(len(B)):
        for j in range(len(A[0])):
            s=0
            for q in range(len(B[0])):
                s+=A[q][j]*B[i][q]
            C[i].append(s)
    return np.array(C)
            











def to_DCT_fast(arr, arr2, two, H, H_rev):              #fast DCT
    
    if(two):                                                    #two arrays
        arr_h=[[] for i in range(math.ceil(len(arr)/8))]
        arr_h2=[[] for i in range(math.ceil(len(arr2)/8))]
        for i in range(len(arr_h)):
            for j in range(math.ceil(len(arr[0])/8)):
                arr_tmp=np.zeros((8, 8))  
                arr_tmp2=np.zeros((8, 8)) 
                for x in range(8):
                    for y in range(8):
                        if(8*i+x < len(arr) and 8*j+y < len(arr[0])):
                            arr_tmp[x][y]=arr[8*i+x][8*j+y]
                            arr_tmp2[x][y]=arr2[8*i+x][8*j+y]
                arr_tmp=arr_tmp-127
                arr_tmp2=arr_tmp2-127
                arr_tmp=mult_matrix(H, arr_tmp)
                arr_tmp2=mult_matrix(H, arr_tmp2)
                arr_tmp=mult_matrix(arr_tmp, H_rev)
                arr_tmp2=mult_matrix(arr_tmp2, H_rev)
                arr_h[i].append(arr_tmp)
                arr_h2[i].append(arr_tmp2)
                
        return arr_h, arr_h2
                
    else:
        arr_h=[[] for i in range(math.ceil(len(arr)/8))]        #one array
        for i in range(len(arr_h)):
            for j in range(math.ceil(len(arr[0])/8)):
                arr_tmp=np.zeros((8, 8))  
                for x in range(8):
                    for y in range(8):
                        if(8*i+x < len(arr) and 8*j+y < len(arr[0])):
                            arr_tmp[x][y]=arr[8*i+x][8*j+y]
                arr_tmp=arr_tmp-127
                #print("\n\n")
                #print(arr_tmp)
                arr_tmp=mult_matrix(H, arr_tmp)
                #print("\n\n")
                #print(arr_tmp)
                arr_tmp=mult_matrix(arr_tmp, H_rev)
                #print("\n\n")
                #print(arr_tmp)
                arr_h[i].append(arr_tmp)

        return arr_h










def reverse_from_DCT_fast(arr_h, arr_h2, two, H, H_rev):         #general reverse DCT cycle       fast

    if(two):
        arr_b=np.zeros((8*len(arr_h), 8*len(arr_h[0])))      #two arrays
        arr_b2=np.zeros((8*len(arr_h2), 8*len(arr_h2[0]))) 
        for i in range(len(arr_h)):
            for j in range(len(arr_h[0])):
                arr_tmp=mult_matrix(H_rev, arr_h[i][j])
                arr_tmp=mult_matrix(arr_tmp, H)
                
                arr_tmp2=mult_matrix(H_rev, arr_h2[i][j])
                arr_tmp2=mult_matrix(arr_tmp2, H)
                
                arr_tmp+=127
                arr_tmp2+=127
                #for k in range(len(arr_tmp)):
                    #for q in range(len(arr_tmp[0])):
                        #arr_tmp[k][q]+=127
                        #arr_tmp2[k][q]+=127

                for x in range(8):
                    for y in range(8):
                        arr_b[i*8+x][j*8+y]=arr_tmp[x][y]
                        arr_b2[i*8+x][j*8+y]=arr_tmp2[x][y]
                        
        return arr_b, arr_b2

        
    else:
        arr_b=np.zeros((8*len(arr_h), 8*len(arr_h[0])))      #one array
        for i in range(len(arr_h)):
            for j in range(len(arr_h[0])):
                arr_tmp=mult_matrix(H_rev, arr_h[i][j])
                arr_tmp=mult_matrix(arr_tmp, H)
                arr_tmp=arr_tmp+127
                
                #for k in range(len(arr_tmp)):
                    #for q in range(len(arr_tmp[0])):
                        #arr_tmp[k][q]+=127
                for x in range(8):
                    for y in range(8):
                        arr_b[i*8+x][j*8+y]=arr_tmp[x][y]
                        
        return arr_b 












def new_quantification(s, Q):               #quantification of given quality
    for i in range(8):
        for j in range(8):
            Q[i][j]=(Q[i][j]*s + 50)/100

    return Q






def quant_matrix(arr, Q):                   #use quantification on DCT
    for i in range(8):
        for j in range(8):
            arr[i][j]=arr[i][j]/Q[i][j]
    arr=np.round(arr)
    arr=np.ndarray.astype(arr, dtype=np.int8)
    return arr




    
def reverse_quant_matrix(arr, Q):               #reverse quantification on DCT
    for i in range(8):
        for j in range(8):
            arr[i][j]=arr[i][j]*Q[i][j]
            
    return arr








    
def to_quant(arr, arr2, two, Q):                #general quantification
    if(two):      
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                arr[i][j]=quant_matrix(arr[i][j], Q)
                arr2[i][j]=quant_matrix(arr2[i][j], Q)
                
        return arr, arr2
    
    else:
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                arr[i][j]=quant_matrix(arr[i][j], Q)
                
        return arr
    





def reverse_from_quant(arr, arr2, two, Q):                                #general reverse quantification
    if(two):
            for i in range(len(arr)):
                for j in range(len(arr[0])):
                    arr[i][j]=reverse_quant_matrix(arr[i][j], Q)
                    arr2[i][j]=reverse_quant_matrix(arr2[i][j], Q)
                    
            return arr, arr2
    
    else:
        for i in range(len(arr)):
                for j in range(len(arr[0])):
                    arr[i][j]=reverse_quant_matrix(arr[i][j], Q)
        return arr









def RLE_plus(line):              #RLE 
    data=bytearray()
    c=0
    tmp=0
    if(line[0]!=0):
        first=True
    else:
        first=False
    for i in range(len(line)):
        if(line[i]==0):
            c+=1
        else:
            data.append(line[i]+127)
            data.append(c)
            c=0
    if(c!=0):
        data.append(127)
        data.append(c-1)
    return data     
            



def to_RLE(arr):                #general use RLE
    data=bytearray()
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            line=matrix_traversal(arr[i][j])
            data=data+RLE_plus(line)
    return data











def put_together_RLE(big_arr, num):          #final look of data
    data=[]
    C=[]
    #if(num>1):
    for i in range(num):         
        data.append(to_RLE(big_arr[i]))
        
        #print("\n\n")
        #print(data[i])
        #print("\n\n")
        C.append(len(data[i]))
    #else:
        #big_arr=matrix_traversal(big_arr[0][0])
        #data.append(RLE_plus(big_arr[0]))
        #C.append(len(data))
        
    data2=bytearray()
    for i in range(num):
        if(C[i]<256):
            data2.append(1)
            data2.append(C[i])
        elif(C[i]<65536):
            data2.append(2)
            y=C[i].to_bytes(2, "big")
            data2.append(int.from_bytes(y[0:1], byteorder="big"))
            data2.append(int.from_bytes(y[1:2], byteorder="big"))
        elif(C[i]<16777215):
            data2.append(3)
            y=C[i].to_bytes(3, "big")
            data2.append(int.from_bytes(y[0:1], byteorder="big"))
            data2.append(int.from_bytes(y[1:2], byteorder="big"))
            data2.append(int.from_bytes(y[2:3], byteorder="big"))
        else:            
            print("\nproblem with len of data\n")
            return
               
        data2+=data[i]

    return data2






def reverse_RLE_plus(data):
    line=[]
    i=0
    while(i<len(data)):
        for j in range(data[i+1]):
            line.append(0)
        line.append(data[i]-127)
        i+=2
    return line






def from_RLE(data):  
    big_line=reverse_RLE_plus(data)
    num_of_arrays=len(big_line)//64
    t=0
    big_arr=[[] for i in range(int(math.sqrt(num_of_arrays)))]
    for i in range(len(big_arr)):
        for j in range(len(big_arr)):
            big_arr[i].append(line_to_matrix(big_line[t*64:t*64+64], 8, 8))
            t+=1

    return big_arr




def reverse_together_RLE(data):
    C=[]
    data2=[]
    data2.append(bytearray())
    i=0
    j=0
    p=0
    while(True):
        k=data[i]
        if(k==1):
            C.append(data[i+1])
            i+=2
        elif(k==2):
            a=data[i+1]
            b=data[i+2]
            a=a<<8
            C.append(a+b)
            i+=3
        else:
            a=data[i+1]
            b=data[i+2]
            e=data[i+3]
            a=a<<16
            b=b<<8
            C.append(a+b+e)
            i+=4
        for t in range(C[j]):
            data2[j].append(data[i+t])
            
        if(len(data)>k+C[j]+1+p):
            i+=C[j]          
            p+=k+C[j]+1
            data2.append(bytearray())
            j+=1
        else:
            break
    
    A=[]
    for i in range(len(C)):
        A.append(from_RLE(data2[i]))
        
    return A





def step_to_data(arr1, arr2, arr3):
    
    A=[]                         
    A.append(arr1)
    A.append(arr2)
    A.append(arr3)
    data=put_together_RLE(A, 3)
    
    file=open("img_out", "wb")
    file.write(data)
    file.close()









def make_H_matrix(T):                       #matrix for DCT
    
    if(T==False):                           #normal
        H=[[] for i in range(8)]
        for i in range(8):
            H[i].append(1/math.sqrt(8))
        for i in range(8):
            for j in range(1,8):
                H[i].append((2/math.sqrt(8))*math.cos(math.pi*((2*i) + 1)*j/16) )

        #print(H)
        return np.array(H)
        
    else:                                   #^-1
        H=[[] for i in range(8)]
        for i in range(8):
            H[0].append(1/math.sqrt(8))
        for i in range(1,8):
            for j in range(8):
                H[i].append( (2/math.sqrt(8))*(math.cos(math.pi*((2*j) + 1)*i/16)) )

        #print(H)
        return np.array(H)









from decimal import Decimal
from re import L
from PIL import Image           #start
import numpy as np
import math
QY=np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92], [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]])
QC=np.array([[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99], [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]])
        




f=0
f2=0
im=0

while(True):        #main cycle    
    print("\nchoose action:\n1 - RGB / YCbCr\n2 - matrix traversal\n3 - downsample\n4 - upsample\n5 - DCT\n6 - quantification\n7 - jpeg\n8 - exit program\n")   #main options
    f = ask(1,8)
    print("")


    if (f==1):      # RGB / YCrCb
        img=open_image()
          
        while(True):        
            print("choose action:\n1 - RGB to YCbCr\n2 - YCbCr to RGB\n3 - show\n4 - save\n5 - exit menu\n")    
            f2=ask(1,5)
        

            if (f2==1):     # RGB to YCbCr
                img=RGB_to_YCbCr(img)
                


            elif(f2==2):        #YCbCr to RGB
                img=YCbCr_to_RGB(img)
                


            elif(f2==3):            #show
                while(True):        
                    print("\nchoose action:\n1 - show Image\n2 - show mode\n3 - show size\n4 - show format\n5 - show bands\n6 - exit\n")
                    im = ask(1,6)
                 
                    if(im==1):
                        img.show()            
                    elif(im==2):
                        print(img.mode)                            
                    elif(im==3):
                        print(img.size)               
                    elif(im==4):
                        print(img.format)               
                    elif(im==5):
                        print(img.getbands())
                    elif(im==6):
                        break
            


            elif(f2==4):        #save
                print("\nenter name\n")
                name=input()
                
                print("\nchoose type:\n1 - RGB\n2 - YCbCr\n3 - RAW\n")
                im=ask(1,3)
                if(im==1):
                    img.save(name)
                    

                elif(im==2):
                    obj=img.load()
                    B=np.zeros([img.height,img.width, 3], dtype=np.uint8)             
                    for i in range(img.height):
                        for j in range(img.width):
                            for c in range (3):
                                B[i,j, c]=obj[j, i][c]
                    file=open(name,"wb")
                    np.save(file, B)
                    file.close()


                elif(im==3):
                    RAW(img)



            elif(f2==5):                  #exit
                break
            







    elif(f==2):         #matrix traversal
        print("\nchoose action:\n1 - traverse array\n2 - traverse array from img\n3 - use RLE on array\n4 - exit menu\n")
        f2=ask(1,4)
        

        if(f2==1):      #make array
            print("\nenter num of rows\n")
            num_rows=int(input())
            print("\nenter num of columns\n")
            num_columns=int(input())
            arr=[[] for i in range(num_columns)]
            c=0
            #print("\nenter values\n")
            for i in range(num_rows * num_columns):
                arr[c].append(i)
                if(i!=0 and (i+1)%num_columns==0):
                    c+=1
               


        elif(f2==2):        #make array from image
            img=open_image()
            arr=make_arr(img,3) 
            print("\nenter channel (from 0 to 2)\n")
            ch=ask(0,2)
            arr=pick_channel(arr, ch)
           


        elif(f2==3):            #make quantization matrix
      
            img=open_image()
            arr=RGB_to_YCbCr(img)
            arr=pick_channel(arr,1)
            arr=to_DCT(arr,0,False)       
            arr=to_quant(arr,0,False, QC)
            
            
            print("\n\n")
            print(arr)
            print("\n\n")

        
        if(f2!=4):              #not exit
    
            if(f2==3):          #RLE
                A=[]
                A.append(arr)
                data=put_together_RLE(A, 1)
                print(data)
                
                A=reverse_together_RLE(data)
                
                print("\n\n")
                print(A[0])
                print("\n\n")

            else:           
                print("\n\n")
                print(arr)
                print("\n\n")
        
                arr=matrix_traversal(arr)  
        
                print("\n\n")
                print(arr)
                print("\n\n")
                
                #arr=line_to_matrix(arr, math.ceil(math.sqrt(len(arr))), math.ceil(math.sqrt(len(arr))))
                
                #print("\n\n")
                #print(arr)
                #print("\n\n")
                
                
                
                    









    elif(f==3):                 #downsample
        img=open_image()
        arr=make_arr(img, 3)
        
        print("\nenter channel (from 0 to 2)\n")
        ch=ask(0,2)
        arr=pick_channel(arr, ch)
        
        print("\nenter downsample multiplier for X\n")
        modX=int(input())
        print("\nenter downsample multiplier for Y\n")
        modY=int(input())
        
        print("\nchoose action:\n1 - delite rows and columns\n2 - replace with average\n3 - replace with closest to average\n")
        f2=ask(1,3)
        

        if(f2==1):    
            arr=down_rows_and_columns(arr, modX, modY)          #rows and columns
            
            
        elif(f2==2):
            arr=down_average(arr, modX, modY)               #average
            
        elif(f2==3):    
            arr=down_next_to_average(arr, modX, modY)           #next to average









    elif(f==4):                     #upsample
        img=open_image()
        arr=make_arr(img, 3)
        
        print("\nenter channel (from 0 to 2)\n")
        ch=ask(0,2)
        arr=pick_channel(arr, ch)
        
        print("\nenter upsample multiplier for X\n")
        modX=int(input())
        print("\nenter upsample multiplier for Y\n")
        modY=int(input())
        
        arr=upsample(arr, modX, modY)







    elif(f==5):         #DCT
        print("\nchoose action:\n1 - DCT from rand array\n2 - from img array\n3 - exit menu\n")
        f2=ask(1,3)
        
        #arr3 = np.zeros((8, 8), dtype=np.double)           #basic array
        #for i in range(8):                        
            #for j in range(8):                     
                #arr[i][j]=(i+1)*(j+1)              
        



        if(f2==1):
            np.random.seed(1)                                  #array int
            arr3=np.random.randint(0,255,(16,16))           
            print("\n\n")
            print(arr3)
            print("\n\n")
            
            arr_3h=to_DCT(arr3, 0, False)                   #code

            arr3_b=reverse_from_DCT(arr_3h, 0, False)       #decode
            arr3_b=np.ndarray.astype(arr3_b, dtype=np.int32)
            print("\n\n")
            print(arr3_b)




        elif(f2==2):                    #img array        double

            img=open_image()
            arr=RGB_to_YCbCr(img)
            arrY=pick_channel(arr, 0)
            arrCb=pick_channel(arr, 1)
            arrCr=pick_channel(arr, 2)

            arr_hY=to_DCT(arrY, 0, False)                       #code
            arr_hCb, arr_hCr=to_DCT(arrCb, arrCr, True)

           
            
        
            arrY_b=reverse_from_DCT(arr_hY, 0, False)                   #decode
            arrCb_b, arrCr_b=reverse_from_DCT(arr_hCb, arr_hCr, True)
            
           

            #img_arr=np.asarray([ [ [arrY_b[j][i], arrCb_b[j][i], arrCr_b[j][i]] for i in range(len(arrY_b))] for j in range(len(arrY_b[0]))])
                
            #img=Image.fromarray(img_arr.astype("uint8"), "YCbCr")
            #img.show() 













    elif(f==6):     #quantification
        print("\nchoose action:\n1 - make standard quantification matrix\n2 - make quantification matrix of given quality\n3 - use quantification matrix on DCT matrix\n4 - exit menu\n")
        f2=ask(1,4)           


        if(f2==1):      #standard quantification matrix
            print("\nquantification matrix for luminance\n")
            print(QY)
            print("\n\nquantification matrix for chrominance")
            print(QC)
            



            
        elif(f2==2):       #quantification matrix of given quality
            
            print("\nenter quality (from 1 to 99)\n")
            while(True):
                q=int(input())
                if(q >0 and q<100):
                    break
                else:
                    print("\nwrong quality\n")
            
            if(q<50):
                s=5000/q
            else:
                s=200-2*q
            QY=new_quantification(s, QY)
            QC=new_quantification(s, QC)
                    
            print("\nquantification matrix for luminance\n")
            print(QY)
            print("\n\nquantification matrix for chrominance")
            print(QC)





        elif(f2==3):        #quantification matrix on DCT            
            arr=np.zeros((8,8))
            for i in range(8):
                for j in range(8):
                    arr[i][j]=(i+1)*(j+1)
            arr=arr-127
            
            print("\n\n")
            print(arr)
            print("\n\n")

            arr=quant_matrix(arr, QC)
            
            print("\n\n")
            print(arr)
            print("\n\n")

            arr=reverse_quant_matrix(arr, QC)
            
            print("\n\n")
            print(arr)
            print("\n\n")






    elif(f==7):                 #jpg
        print("\nchoose action:\n1 - code\n2 - decode\n")
        f2=ask(1,2)
        if(f2==1):
            
            img=open_image()   
            arr=RGB_to_YCbCr(img)      #code         to YCbCr
                       
        
            Y=pick_channel(arr, 0)
            Cb=pick_channel(arr, 1)
            Cr=pick_channel(arr, 2)
           
                       
            Cb=down_average(Cb, 2, 2)       #downsample
            Cr=down_average(Cr, 2, 2)
                  
            
        
            hY=to_DCT(Y, 0, False)                       #DCT
            hCb, hCr=to_DCT(Cb, Cr, True)


            #H=make_H_matrix(False)                   #DCT fast
            #H_rev=make_H_matrix(True)
            #hY=to_DCT_fast(Y,0, False, H, H_rev)
            #hCb, hCr=to_DCT_fast(Cb,Cr, True, H, H_rev)
            
            #print("\n\n")
            #print(hY)
            #print("\n\n")
            #print(hCb)
            #print("\n\n")
            #print(hCr)


            print("\nenter quality (from 1 to 99)\n")          #get quant matrix
            while(True):
                q=int(input())
                if(q >0 and q<100):
                    break
                else:
                    print("\nwrong quality\n")
            
            if(q<50):
                s=5000/q
            else:
                s=200-2*q
            QY=new_quantification(s, QY)
            QC=new_quantification(s, QC)

            hY=to_quant(hY, 0, False, QY)                #quantification
            hCb, hCr=to_quant(hCb, hCr, True, QC)
        
            


                
            A=[]                            #RLE
            A.append(hY)
            A.append(hCb)
            A.append(hCr)
            data=put_together_RLE(A, 3)
            
            file=open("img_out", "wb")
            file.write(data)
            file.close()
        

        

        else:       #decode
            file=open("img_out", "rb")
            data=file.read()
            file.close()
            A=reverse_together_RLE(data)        #reverse RLE
        
            hY=A[0]
            hCb=A[1]
            hCr=A[2]
        


       


            hY=reverse_from_quant(hY, 0, False, QY)              #reverse quantification
            hCb, hCr=reverse_from_quant(hCb, hCr, True, QC)
        



         

            Y=reverse_from_DCT(hY, 0, False)                   #reverse DCT
            Cb, Cr=reverse_from_DCT(hCb, hCr, True)


            #H=make_H_matrix(False)                        #reverse DCT fast
            #H_rev=make_H_matrix(True)
            
            #Y=reverse_from_DCT_fast(hY, 0, False, H, H_rev)                
            #Cb, Cr=reverse_from_DCT_fast(hCb, hCr, True, H, H_rev)
            
            
      

            Cb=upsample(Cb, 2, 2)           #upsample
            Cr=upsample(Cr, 2, 2)
            print("\n\n")
            print(len(Y))
            print(len(Y[0]))
            print("\n\n")
            print(len(Cb))
            print(len(Cb[0]))
            print("\n\n")
            print(len(Cr))
            print(len(Cr[0]))

            img_arr=np.asarray([ [ [Y[j][i], Cb[j][i], Cr[j][i]] for i in range(len(Y))] for j in range(len(Y[0]))])        #make image
                
            img=Image.fromarray(img_arr.astype("uint8"), "YCbCr")
            img.show() 
        

        
        


       




    elif(f==8):     #exit program   
        break
    
