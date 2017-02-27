function img = readImage( img_path )

% open input image
img = imread(img_path);
img = imresize(img, 0.5);   
%converting to grayscale
if size(img,3)==3
img=rgb2gray(img);
end
% convert from uint8 to double
img=double(img);
