function F=extract_proplery(img)


red=img(:,:,1);
red=reshape(red,1,[]);

g=img(:,:,2);
g=reshape(g,1,[]);

b=img(:,:,3);
b=reshape(b,1,[]);
average_red=mean(red); 
average_g=mean(g); 
average_b=mean(b); 
F=[average_red  average_g  average_b]; 
% Returns a row [rand rand .... rand] representing an image descriptor
% computed from image 'img'

% Note img is a normalised RGB image i.e. colours range [0,1] not [0,255].

return;