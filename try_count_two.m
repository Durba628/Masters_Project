clc; clear; close all;

img = imread("C:\Users\durba\Downloads\Durba_PM\Durba_PM\Agg\9 (17)_Export_RGB_FITC.tif");  
imshow(img)
title('Original Image');

I_crop = imcrop(img);
figure;
imshow(I_crop)
title('Cropped image (removed white scale bar)');

while true
    [x, y, button] = ginput(1);

    if isempty(button)
        break;
    end

    x = round(x);
    y = round(y);

    pixelValue = I_crop(y, x, :);
    fprintf('Pixel (%d,%d): ', x, y);
    disp(pixelValue);
end

mv=input("Enter threshold pixel intensity value for binarization: ");


G = I_crop(:,:,2);

figure; imshow(G, []);
title('Green Channel');


% se = strel('disk', 15);   
% G_tophat = imtophat(G, se);
% 
% figure; imshow(G_tophat, []);
% title('Top-hat filtered (Green)');


BW = G > mv;

figure; imshow(BW);
title('Thresholded (Green > 200)');

BW = bwareaopen(BW, 5);

CC = bwconncomp(BW);
numObjects = CC.NumObjects;

stats = regionprops(CC, 'BoundingBox');

figure;
imshow(G, []); 
hold on;
title(['Detected Aggregates: ', num2str(numObjects)]);

for i = 1:numObjects
    rectangle('Position', stats(i).BoundingBox, 'EdgeColor', 'r', 'LineWidth', 1);
end
hold off;
