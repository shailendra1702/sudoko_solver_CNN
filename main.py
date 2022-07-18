import cv2
import numpy as np
from scipy import stats
from tensorflow import keras
from PIL import Image, ImageDraw, ImageFont

model = keras.models.load_model('model')
font = ImageFont.truetype("./arial.ttf", 56)
grid_size = (60 + 8)*9

def draw_grid(grid):
    grid_img = Image.new('RGBA', (grid_size, grid_size))
    draw = ImageDraw.Draw(grid_img)
    draw.rectangle((0, 0, grid_size, grid_size), fill="#ffffff")
    for i in range(0, grid_size, 68 * 3):
        draw.line((0, i, grid_size, i), fill="#000000", width=5)
        draw.line((i, 0, i, grid_size), fill="#000000", width=5)
    for i in range(0, grid_size, 68):
        draw.line((0, i, grid_size, i), fill="#000000", width=1)
        draw.line((i, 0, i, grid_size), fill="#000000", width=1)
    
    for i in range(9):
        for j in range(9):
            draw.text((j * 68 + 20, i * 68), str(grid[i][j]), fill="#000000", font=font, align="left", )
    grid_img.show()

def is_valid_sudoku_grid(grid):
    def notInRow(arr, row):
        st = set()
    
        for i in range(0, 9):
            if arr[row][i] in st:
                return False

            if arr[row][i] != 0:
                st.add(arr[row][i])
        
        return True
    
    def notInCol(arr, col):
        st = set()
        for i in range(0, 9):
            if arr[i][col] in st:
                return False

            if arr[i][col] != 0:
                st.add(arr[i][col])
        
        return True
    
    def notInBox(arr, startRow, startCol):
        st = set()
        for row in range(0, 3):
            for col in range(0, 3):
                curr = arr[row + startRow][col + startCol]
                if curr in st:
                    return False

                if curr != 0:
                    st.add(curr)

        return True

    def isValid(arr, row, col):
        return (notInRow(arr, row) and notInCol(arr, col) and
                notInBox(arr, row - row % 3, col - col % 3))

    for i in range(0, 9):
        for j in range(0, 9):
            if not isValid(grid, i, j):
                return False
         
    return True

def isSafe(grid, row, col, num):
    for x in range(9):
        if grid[row][x] == num:
            return False
            
    for x in range(9):
        if grid[x][col] == num:
            return False
            
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True

def solveSudoku(grid, row, col):
    N = 9
    if (row == N - 1 and col == N):
        return True
    if col == N:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return solveSudoku(grid, row, col + 1)
    for num in range(1, N + 1, 1):
        if isSafe(grid, row, col, num):
            grid[row][col] = num
            if solveSudoku(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False

def main():
    cap = cv2.VideoCapture(0)
    t = 0
    memoize = [[[] for _ in range(9)] for _ in range(9)]
    print(memoize)
    while True:
        success, frame = cap.read()

        if not success:
            print("[ERR] Video capture failed to give a frame")
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_denoise = cv2.GaussianBlur(img, (5,5), 3)
        img_edge = cv2.Canny(img_denoise, 50, 50)

        contours, _ = cv2.findContours(
            img_edge, 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        for contour in contours:
            c_area = cv2.contourArea(contour)
            if c_area > 20000:
                cv2.drawContours(frame, contour, -1, (0, 255, 0), 2)
                perimeter = cv2.arcLength(contour, True)
                polygon_approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

                p1 = np.array([polygon_approximation.item(0), polygon_approximation.item(1)])
                p2 = np.array([polygon_approximation.item(2), polygon_approximation.item(3)])
                p3 = np.array([polygon_approximation.item(4), polygon_approximation.item(5)])
                p4 = np.array([polygon_approximation.item(6), polygon_approximation.item(7)])

                cv2.circle(frame, p1, 10, (255, 0, 0), 2)
                cv2.circle(frame, p2, 10, (0, 255, 0), 2)
                cv2.circle(frame, p3, 10, (0, 0, 255), 2)
                cv2.circle(frame, p4, 10, (12, 55, 0), 2)

                img_coords = np.float32([p1, p2, p3, p4])
                map_coords = np.float32(
                   [[grid_size, 0], [0, 0], [0, grid_size], [grid_size, grid_size]]
                )

                transform_matrix = cv2.getPerspectiveTransform(img_coords, map_coords)
                extracted_grid = cv2.warpPerspective(img, transform_matrix, (grid_size, grid_size))
                extracted_grid_th = cv2.adaptiveThreshold(extracted_grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
                extracted_grid_th = cv2.erode(extracted_grid_th, np.ones((3,3)))
                extracted_grid_th = cv2.dilate(extracted_grid_th, np.ones((3,3)))
                extracted_grid_th = cv2.erode(extracted_grid_th, np.ones((3,3)))
                extracted_digit_imgs = []
                for y in range(0, 9):
                    for x in range(0, 9):
                        digit_img = extracted_grid_th[68 * x + 4:68 * x + 64, 68 * y + 4:68 * y + 64]
                        digit_img = 255 - digit_img
                        digit_img = digit_img[5:55, 5:55]
                        digit_img = np.pad(digit_img, pad_width=5, mode='constant', constant_values=0)
                        extracted_digit_imgs.append(digit_img)
                
                # if t < 6000:
                #     cv2.imwrite(f'./data2/0/{t}.png', extracted_digit_imgs[0 + 9 * 2])
                #     cv2.imwrite(f'./data2/1/{t}.png', extracted_digit_imgs[1 + 9 * 3])
                #     cv2.imwrite(f'./data2/2/{t}.png', extracted_digit_imgs[5 + 9 * 4])
                #     cv2.imwrite(f'./data2/3/{t}.png', extracted_digit_imgs[0 + 9 * 1])
                #     cv2.imwrite(f'./data2/4/{t}.png', extracted_digit_imgs[4 + 9 * 0])
                #     cv2.imwrite(f'./data2/5/{t}.png', extracted_digit_imgs[0 + 9 * 0])
                #     cv2.imwrite(f'./data2/6/{t}.png', extracted_digit_imgs[1 + 9 * 0])
                #     cv2.imwrite(f'./data2/7/{t}.png', extracted_digit_imgs[5 + 9 * 0])
                #     cv2.imwrite(f'./data2/8/{t}.png', extracted_digit_imgs[3 + 9 * 0])
                #     cv2.imwrite(f'./data2/9/{t}.png', extracted_digit_imgs[2 + 9 * 1])
                #     t += 1

                # if t == 6000:
                #     print('DONE')

                extracted_digit_imgs = np.float32(extracted_digit_imgs)
                extracted_digit_imgs /= 255
                extracted_digit_imgs = extracted_digit_imgs.reshape(extracted_digit_imgs.shape[0], 60, 60, 1)

                grid = np.array([x.argmax() for x in model.predict(extracted_digit_imgs)]).reshape(9, 9).T
                if t < 5000:
                    for i in range(9):
                        for j in range(9):
                            memoize[i][j].append(grid[i][j])
                    t += 1
                
                if t == 5000:
                    for i in range(9):
                        for j in range(9):
                            grid[i][j] = stats.mode(memoize[i][j])[0][0]
                memoize = [[[] for _ in range(9)] for _ in range(9)]
                if is_valid_sudoku_grid(grid):
                    if solveSudoku(grid, 0, 0):
                        print(grid)
                        draw_grid(grid)

                cv2.imshow("Detected", extracted_grid_th)

        
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()