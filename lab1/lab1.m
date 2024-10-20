img = imread("testimages/sphinx.jpg");
% img(200:300,100:150,:) = 255;
normimg = double(img)./255; 
grayscale = normimg(:, :, 1)*0.3 + normimg(:, :, 2)*0.59 + normimg(:, :, 3)*0.11;
grayscale2 = normimg(:, :, 1)*0.33 + normimg(:, :, 2)*0.33 + normimg(:, :, 3)*0.33;

% figure;
% tiledlayout(2,1)
% nexttile
% imshow(grayscale)
% title("nice")
% 
% nexttile
% imshow(grayscale2)
% title("bad")

% K = ones(5,5)  ./25
% blurred = conv2(grayscale, K, 'same')
% imshow(blurred)

Kx = [1 2 1; 0 0 0; -1 -2 -1];
Ky = Kx';
dx = conv2(grayscale, Kx, 'same');
dy = conv2(grayscale, Ky, 'same');
mag = sqrt(dx.^2 + dy.^2);
hist = histogram(mag);
thresholded = (mag >0.32);
imshow(thresholded)