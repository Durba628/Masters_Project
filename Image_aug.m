img_folder = "D:\Count My Proteins\Each_augmentation_folder\Original images";
imds = imageDatastore(img_folder);
oF = imds.Files;

v  = "D:\Count My Proteins\yolo_train_dataset\images\val";
test = "D:\Count My Proteins\Each_augmentation_folder\orig_test";
train = "D:\Count My Proteins\Each_augmentation_folder\orig_train";

if ~exist(v,  'dir'), mkdir(v);  end
if ~exist(test, 'dir'), mkdir(test); end
if ~exist(train,'dir'), mkdir(train);end

val_count   = 0;
test_count  = 0;
train_count = 0;

for j = 1:numel(oF)
    n=imread(oF{j});
    
    
    if mod(j, 9) == 0
       
        newName = sprintf("val_img%d.png", j);
        savePath = fullfile(v, newName);
        imwrite(n, savePath);
        val_count = val_count + 1;

    elseif mod(j, 9) == 1 && j > 1
        
        newName = sprintf("test_img%d.png", j);
        savePath = fullfile(test, newName);
        imwrite(n, savePath);
        test_count = test_count + 1;

    else
        
        newName = sprintf("train_img%d.png", j);
        savePath = fullfile(train, newName);
        imwrite(n, savePath);
        train_count = train_count + 1;
    end

    if mod(j, 1000) == 0
        fprintf("Processed %d / %d\n", j, numel(oF));
    end
end

fprintf("\nSplit complete.\n");
fprintf("Train : %d images (%.1f%%)\n", train_count, 100*train_count/numel(oF));
fprintf("Val   : %d images (%.1f%%)\n", val_count,   100*val_count/numel(oF));
fprintf("Test  : %d images (%.1f%%)\n", test_count,  100*test_count/numel(oF));
fprintf("Total : %d images\n", train_count + val_count + test_count);
%%
%460, 91, 369

img_folder="D:\Count My Proteins\Each_augmentation_folder\orig_train";

imds = imageDatastore(img_folder);
oF = imds.Files;
for j=1:numel(oF)
    [~, fname, ext] = fileparts(oF{j});
    n=imread(oF{j});
    d=(size(n));
    r=d(1);
    c=d(2);
   
    r_m=floor(r/2);
    c_m=floor(c/2);
    
    %ORIGINAL 92 images
    copyfile(oF{j}, fullfile("D:\Count My Proteins\Each_augmentation_folder\aug_imgs_4parts", [fname ext]));
    
    
   
    %TOP-LEFT
    i_tl=n(1:r_m,1:c_m,:);
    name=sprintf("tl_img%d.png",j);
    
    file_path_2=fullfile("D:\Count My Proteins\Each_augmentation_folder\aug_imgs_4parts",name);
    
    imwrite(i_tl, file_path_2);
    
    %TOP-RIGHT
    i_tr=n(1:r_m,c_m+1:end,:);
    name=sprintf("tr_img%d.png",j);
    
    file_path_2=fullfile("D:\Count My Proteins\Each_augmentation_folder\aug_imgs_4parts",name);
    
    imwrite(i_tr, file_path_2);
    
    %BOTTOM-LEFT
    i_bl=n(r_m+1:end,1:c_m,:);
    name=sprintf("bl_img%d.png",j);
    
    file_path_2=fullfile("D:\Count My Proteins\Each_augmentation_folder\aug_imgs_4parts",name);
    
    imwrite(i_bl, file_path_2);
    
    %BOTTOM-RIGHT
    i_br=n(r_m+1:end,c_m+1:end,:);
    name=sprintf("br_img%d.png",j);
    
    file_path_2=fullfile("D:\Count My Proteins\Each_augmentation_folder\aug_imgs_4parts",name);
    
    imwrite(i_br, file_path_2);
end
%%
img_folder = "D:\Count My Proteins\Each_augmentation_folder\aug_imgs_4parts\non_blanks";
imds = imageDatastore(img_folder);
oF = imds.Files;



angles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350];


for j = 1:numel(oF)

    n = imread(oF{j});
    name = sprintf("orig_0°img%d.png", j);
    file_path = fullfile("D:\Count My Proteins\Each_augmentation_folder\aug_imgs_angles", name);
    imwrite(n, file_path);
end

% Rotate at all angles
for j = 1:numel(oF)
    n = imread(oF{j});
    for k = 1:numel(angles)
        ang = angles(k);
        rotated = imrotate(n, ang, 'crop');
        name = sprintf("%d°_img%d.png", ang, j);
        file_path = fullfile("D:\Count My Proteins\Each_augmentation_folder\aug_imgs_angles", name);
        imwrite(rotated, file_path);
    end
end

fprintf("Augmentation complete.\n");
fprintf("Total images: %d originals + %d rotated = %d total\n", numel(oF), numel(oF)*numel(angles), numel(oF)*(1 + numel(angles)));

%%
% label generation of angles
j=0;

imds = imageDatastore("D:\Count My Proteins\Each_augmentation_folder\aug_imgs_angles");
oF = imds.Files;
for i=1:numel(oF)
     n=imread(oF{i});

     if (j==0)
         n_1=imread(oF{1});
         figure;
         imshow(n_1);
         while true
            [x, y, button] = ginput(1);

            if isempty(button)
                break;
            end

            x = round(x);
            y = round(y);

            pixelValue = n_1(y, x, :);
            fprintf('Pixel (%d,%d): ', x, y);
            disp(pixelValue);
        end

         mv=input("Enter threshold pixel intensity value for binarization: ");
         j=1;
     end
     R=n(:,:,1);
     G=n(:,:,2);
     B=n(:,:,3);

     BW = R < mv & G > mv & B < mv;

     

     BW = bwareaopen(BW, 5);
     imgHeight = size(BW, 1);
     imgWidth  = size(BW, 2);
     CC = bwconncomp(BW);
     numObjects = CC.NumObjects;

     stats = regionprops(CC, 'BoundingBox');
     img_annot = n;  
     for m = 1:length(stats)
         bb = stats(m).BoundingBox;
         img_annot = insertShape(img_annot, 'Rectangle', bb, 'Color', 'white', 'LineWidth', 2);
     end
     [~, name, ~] = fileparts(oF{i});
     labelFile = fullfile("D:\Count My Proteins\All_labels", strcat(name, '.txt'));
     fid = fopen(labelFile, 'w');
     for m = 1:numObjects
        pixels = CC.PixelIdxList{m};
        [y, x] = ind2sub([imgHeight, imgWidth], pixels);

        x_center = mean(x)/imgWidth;
        y_center = mean(y)/imgHeight;
        width    = (max(x) - min(x))/imgWidth;
        height   = (max(y) - min(y))/imgHeight;

        fprintf(fid, '0 %.6f %.6f %.6f %.6f\n', x_center, y_center, width, height);
     end
     fclose(fid);
     f=imds.Files{i};


     
     [~, name, ~] = fileparts(oF{i});
     savePath = fullfile("D:\Count My Proteins\All_images", strcat(name, '.png'));
     imwrite(n, savePath);
     
     [~, name, ~] = fileparts(oF{i});
     savePath = fullfile("D:\Count My Proteins\Ground-truth", strcat(name, '.png'));
     imwrite(img_annot, savePath);
     
end
%%

clc; clear; close all;
%%

parpool('local', 6); % 4–8 workers 
%% 
imgFolder     = "C:\Count My Proteins\Each_augmentation_folder\aug_imgs_angles";
labelFolder   = "C:\Count My Proteins\All_labels";
allImagesFolder = "C:\Count My Proteins\All_images";
intensityFolder = "C:\Count My Proteins\Each_augmentation_folder\Intensity_Images";
grayFolder      = "C:\Count My Proteins\Each_augmentation_folder\gray_Images";
gtFolder        = "C:\Count My Proteins\Ground-truth";
noisyFolder     = "C:\Count My Proteins\Each_augmentation_folder\noisy_images";

folders = [allImagesFolder, intensityFolder, grayFolder, gtFolder, noisyFolder];
for f = folders
    if ~exist(f, 'dir'), mkdir(f); end
end


imds = imageDatastore(imgFolder);
files = imds.Files;
N = numel(files);

fprintf('Total images: %d\n', N);


allData = cell(N,1);
validMask = false(N,1);

for i = 1:N
    [~, baseName, ~] = fileparts(files{i});
    lp = fullfile(labelFolder, baseName + ".txt");

    if exist(lp,'file')
        allData{i} = readmatrix(lp);
        validMask(i) = true;
    end
end

validIdx = find(validMask);
fprintf('Valid images: %d\n', numel(validIdx));


multiplier_range = [0.6, 1.4];
q1 = multiplier_range(1) + 0.25 * (multiplier_range(2) - multiplier_range(1));
q4 = multiplier_range(2);

multipliers    = [q1, q4];
quartile_names = ["q1", "q4"];

noise_fraction = 0.15;
noise_variance = 0.025;

rng(42);
numToNoise = round(noise_fraction * numel(validIdx));
noiseIdx   = validIdx(randperm(numel(validIdx), numToNoise));


if isempty(gcp('nocreate'))
    parpool('local', 6);
end

fprintf('Starting fast parallel processing...\n');

parfor idx = 1:numel(validIdx)

    i = validIdx(idx);

    img = imread(files{i});
    img_single = im2single(img);
    data = allData{i};

    [h,w,~] = size(img);

    % Convert boxes once
    if ~isempty(data)
        boxes = data(:,2:5);
        pBoxes = zeros(size(boxes));
        for b = 1:size(boxes,1)
            xc = boxes(b,1)*w; yc = boxes(b,2)*h;
            bw = boxes(b,3)*w; bh = boxes(b,4)*h;
            pBoxes(b,:) = [xc-bw/2, yc-bh/2, bw, bh];
        end
    else
        pBoxes = [];
    end

    for k = 1:numel(multipliers)

        adj = min(max(img_single * multipliers(k),0),1);
        img_out = im2uint8(adj);

        if ~isempty(pBoxes)
            img_gt = insertShape(img_out,'Rectangle',pBoxes,...
                'Color','white','LineWidth',2);
        else
            img_gt = img_out;
        end

        newBase = sprintf("%s_img%d", quartile_names(k), i);

        imwrite(img_out, fullfile(allImagesFolder, newBase + ".png"));
        imwrite(img_out, fullfile(intensityFolder, newBase + ".png"));
        imwrite(img_gt,  fullfile(gtFolder, newBase + ".png"));
        writematrix(data, fullfile(labelFolder, newBase + ".txt"));
    end

 
    grayImg = rgb2gray(img);

    if ~isempty(pBoxes)
        grayGT = insertShape(grayImg,'Rectangle',pBoxes,...
            'Color','white','LineWidth',2);
    else
        grayGT = grayImg;
    end

    newBase = sprintf("gray_img%d", i);

    imwrite(grayImg, fullfile(allImagesFolder, newBase + ".png"));
    imwrite(grayImg, fullfile(grayFolder, newBase + ".png"));
    imwrite(grayGT,  fullfile(gtFolder, newBase + ".png"));
    writematrix(data, fullfile(labelFolder, newBase + ".txt"));

    
    if ismember(i, noiseIdx)

        noise = randn(size(img_single),'single') * sqrt(noise_variance);
        noisy = min(max(img_single + noise,0),1);
        noisyImg = im2uint8(noisy);

        if ~isempty(pBoxes)
            noisyGT = insertShape(noisyImg,'Rectangle',pBoxes,...
                'Color','white','LineWidth',2);
        else
            noisyGT = noisyImg;
        end

        newBase = sprintf("noisy_img%d", i);

        imwrite(noisyImg, fullfile(allImagesFolder, newBase + ".png"));
        imwrite(noisyImg, fullfile(noisyFolder, newBase + ".png"));
        imwrite(noisyGT,  fullfile(gtFolder, newBase + ".png"));
        writematrix(data, fullfile(labelFolder, newBase + ".txt"));
    end

end

fprintf('ALL AUGMENTATIONS COMPLETE.\n');
%%
