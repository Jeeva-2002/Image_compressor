# Required libraries
import base64
from io import BytesIO
from PIL import Image
import streamlit as st
from math import sqrt, cos
import math
import numpy as np
import cv2

# Luninance matrix for DCT
Q = np.array([[80, 60, 50, 80, 120, 200, 255, 255],
              [55, 60, 70, 95, 130, 255, 255, 255],
              [70, 65, 80, 120, 200, 255, 255, 255],
              [70, 85, 110, 145, 255, 255, 255, 255],
              [90, 110, 185, 255, 255, 255, 255, 255],
              [120, 175, 255, 255, 255, 255, 255, 255],
              [245, 255, 255, 255, 255, 255, 255, 255],
              [255, 255, 255, 255, 255, 255, 255, 255]])


# Function to download BMP files
def get_image_download_link(img):
    im = Image.fromarray(img)
    buffered = BytesIO()
    im.save(buffered, format="BMP")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/bmp;base64,{img_str}">Download result</a>'
    return href


# Function to download JPEG files
def get_image_download_link1(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpeg;base64,{img_str}">Download result</a>'
    return href


# Function to calculate distance
def distance(x, y):
    ans = 0
    for i in range(3):
        ans += (x[i] - y[i]) ** 2
    return ans ** 0.5


# Function to calculate distortion
def distortion(y1, y2, v1, v2):
    ans = 0
    for i in v1:
        ans += math.ceil(distance(i, y1) ** 2)
    for i in v2:
        ans += math.ceil(distance(i, y2) ** 2)
    ans = ans / (len(v1) + len(v2))
    return ans


# Function for DCT conversion
def DCT(mat):
    pi = 3.142857
    dct = []
    for i in range(0, 8):
        dct_row = []
        for j in range(0, 8):
            if i == 0:
                ci = 1 / sqrt(8)
            else:
                ci = sqrt(2) / sqrt(8)
            if j == 0:
                cj = 1 / sqrt(8)
            else:
                cj = sqrt(2) / sqrt(8)
            sum = 0
            for k in range(0, 8):
                for l in range(0, 8):
                    dct1 = mat[k][l] * cos((2 * k + 1) * i * pi / (2 * 8)) * cos((2 * l + 1) * j * pi / (2 * 8))
                    sum = sum + dct1
            dct_row.append(ci * cj * sum)
        dct.append(dct_row)
    return np.array(dct)


# Function for the Uniform quantization
@st.cache
def Quant(img, n):
    imgGray = img.convert('L')
    array = list(imgGray.getdata())
    div = 2 ** n
    x = range(256)

    l = np.array_split(np.array(x), div)

    for j in range(len(array)):
        for i in l:
            if i[0] <= array[j] <= i[len(i) - 1]:
                array[j] = (((i[len(i) - 1] - i[0]) // 2) + i[0])

    return array

# Function for LBG compression
@st.cache
def LBG(arr):
    y1 = list(arr[0])
    y2 = list(arr[1])
    err = 0.02

    D0 = 0

    v1 = []
    v2 = []

    boolean = False
    while not boolean:
        v1 = []
        v2 = []
        for i in range(len(arr)):
            a1 = distance(y1, arr[i])
            a2 = distance(y2, arr[i])
            if a1 < a2:
                v1.append(arr[i])
            else:
                v2.append(arr[i])

        D1 = distortion(y1, y1, v1, v2)
        boolean = abs((D1 - D0)) / D1 < err
        if not boolean:
            if len(v1) > 0:
                for i in range(3):
                    val = 0
                    for j in range(len(v1)):
                        val += v1[j][i]
                    y1[i] = math.ceil(val / len(v1))
            if len(v2) > 0:
                for i in range(3):
                    val = 0
                    for j in range(len(v2)):
                        val += v2[j][i]
                    y2[i] = math.ceil(val / len(v2))

        D0 = D1

    for i in range(len(v1)):
        arr[arr.index(v1[i])] = tuple(y1)
    for i in range(len(v2)):
        arr[arr.index(v2[i])] = tuple(y2)

    return arr


# Function for Llyod-Max compression
@st.cache
def llyodMax(x_arr):
    epsilon = 0.02
    Dold = 0
    split_val = 128
    min_arr_sub_val = 64
    max_arr_sub_val = 192
    flag = False
    Iteration = 0
    while not flag:
        first_arr = []
        second_arr = []
        for x in x_arr:
            if x < split_val:
                first_arr.append(x)
            else:
                second_arr.append(x)

        sq_less_arr = [(x - min_arr_sub_val) ** 2 for x in first_arr]
        sq_more_arr = [(y - max_arr_sub_val) ** 2 for y in second_arr]

        Dnew = sum(sq_less_arr) + sum(sq_more_arr)
        if abs(Dnew - Dold) < epsilon:
            flag = True
            continue
        Dold = Dnew
        if len(first_arr) > 0:
            min_arr_sub_val = math.ceil(sum(first_arr) / len(first_arr))

        else:
            min_arr_sub_val = 0
        if len(second_arr) > 0:
            max_arr_sub_val = math.ceil(sum(second_arr) / len(second_arr))
        else:
            max_arr_sub_val = 0
        split_val = (min_arr_sub_val + max_arr_sub_val) // 2

    low = math.ceil(sum(first_arr) / len(first_arr))
    high = math.ceil(sum(second_arr) / len(second_arr))
    return high, low


# Function for Jpeg Image Compression
@st.cache
def dctmain(img):
    height = len(img)
    width = len(img[0])
    slicedimg = []
    block = 8
    currY = 0  # current Y index
    for i in range(block, height + 1, block):
        currX = 0  # current X index
        for j in range(block, width + 1, block):
            slicedimg.append(img[currY:i, currX:j] - np.ones((8, 8)) * 128)  # Extracting 128 from all pixels
            currX = j
        currY = i
    imf = [np.float32(img) for img in slicedimg]
    DCToutput = []
    for part in imf:
        currDCT = DCT(part)
        DCToutput.append(currDCT)
    row = 0
    rowNcol = []
    for j in range(int(width / block), len(DCToutput) + 1, int(width / block)):
        rowNcol.append(np.hstack((DCToutput[row:j])))
        row = j
    re1 = np.vstack(rowNcol)
    for dct in DCToutput:
        for i in range(block):
            for j in range(block):
                dct[i, j] = np.around(dct[i, j] / Q[i, j])
    for dctinv in DCToutput:
        for i in range(block):
            for j in range(block):
                dctinv[i, j] = np.around(dctinv[i, j] * Q[i, j])
    inv = []
    for dct in DCToutput:
        curriDCT = cv2.idct(dct)
        inv.append(curriDCT)
    row = 0
    rowNcol = []
    for j in range(int(width / block), len(inv) + 1, int(width / block)):
        rowNcol.append(np.hstack((inv[row:j])))
        row = j
    res = np.vstack(rowNcol)
    res = res.astype(int)
    return res


def main():
    st.title("Online Image Compressor")

    menu = ["Uniform Quantization", "LBG Quantization", "Llyod-Max Quantization", "JPEG Compression"]
    choice = st.sidebar.selectbox("Menu", menu)


    if choice == "Uniform Quantization":
        st.subheader("Uniform Quantization")
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        int_val = st.slider('Choose the Quantization bits...', min_value=1, max_value=7, value=3, step=1)

        if st.button('Convert'):
            image = Image.open(uploaded_file)
            pixels = Quant(image, int_val)
            res = Image.new('L', image.size)
            res.putdata(pixels)
            st.image(res, caption='Compressed Image')
            st.markdown(get_image_download_link1(res), unsafe_allow_html=True)

    elif choice == "LBG Quantization":
        st.subheader("LBG Quantization")
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        if st.button('Convert'):
            image = Image.open(uploaded_file)
            arr = list(image.getdata())
            array = LBG(arr)
            res = Image.new("RGB", image.size)
            res.putdata(array)
            st.image(res, caption='Compressed Image')
            st.markdown(get_image_download_link1(res), unsafe_allow_html=True)

    elif choice == "Llyod-Max Quantization":
        st.subheader("Llyod-Max Quantization")
        uploaded_file = st.file_uploader("Choose an image...", type="bmp")
        if st.button('Convert') and uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 0)
            x_arr = []
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    x_arr.append(img[i][j])

            high, low = llyodMax(x_arr)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i][j] <= 76:
                        img[i][j] = low
                    elif img[i][j] >= 77:
                        img[i][j] = high
            st.image(img, caption='Compressed Image')
            st.markdown(get_image_download_link(img), unsafe_allow_html=True)

    elif choice == "JPEG Compression":
        st.subheader("JPEG Compression")
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        if st.button('Convert') and uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 0)
            res = dctmain(img)
            st.image(res, caption='Compressed Image', clamp=True)
            im = Image.fromarray(res, 'RGB')
            st.markdown(get_image_download_link1(im), unsafe_allow_html=True)


if __name__ == '__main__':
    main()
