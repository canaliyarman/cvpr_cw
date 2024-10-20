% ex1-2
% pt = rand(2,5000); 
% pt(3,:)=0;
% pt(2,:)=pt(2,:) * 5;
% xlabel ('X axis'); 
% ylabel ('Y axis'); 
% zlabel ('Z axis'); 
% plot3(pt(1,:), pt(2,:), pt(3,:), 'rx');
% e = Eigen_Build(pt);    
% org = [mean(pt(1,:)); mean(pt(2,:)); mean(pt(3,:))];
% ptsub = pt-org;
% c = (ptsub*ptsub')./5000;
% [vct val] = eig(c); 

%ex 3
target=double(imread('./testimages/target_peas.bmp'))./255; 
target_obs=[reshape(target(:,:,1),1,[])  ;  reshape(target(:,:,2),1,[])  ;   
reshape(target(:,:,3),1,[]) ]; 
target_e = Eigen_Build(target_obs); 
test = double(imread('./testimages/kitchen.bmp'))./255; 
test_obs=[reshape(test(:,:,1),1,[])  ;  reshape(test(:,:,2),1,[]);   
reshape(test(:,:,3),1,[]) ];
target_e = Eigen_Build(target_obs); 
mdist =  Eigen_Mahalanobis(test_obs,target_e);
result = reshape(mdist,size(test,1),size(test,2)); 
nresult=result./ max(max(result));
imgshow(nresult); 
% imgshow (result < 3); 
% x=test_obs(:,1);
% xsub = x-target_e.org; 
% V= diag(target_e.val);
% U= target_e.vct;
% mdist_squared = xsub' * U * inv(V) * U' * xsub; 