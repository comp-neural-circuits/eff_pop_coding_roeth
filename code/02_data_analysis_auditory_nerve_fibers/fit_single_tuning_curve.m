tuning_curve = @(A,xdata)A(1)+(A(2)-A(1))*erfc((A(3)-xdata)/sqrt(2)/A(4))/2;
file_path = '..\AMT Data\mat_files\';    %put AMT data files here
file = dir([file_path,'*.mat']);
mkdir('single_tuning_curves')
A0 = [0, 300, 40, 5];
A = zeros(length(file),4);
lb = [0, 0, 0, 0];

n = 1;

for k = 1:length(file)
       s = [];
       fr = [];
       load([file_path,file(k).name]);
       for j = 1:length(data)
               if  sum(data{1,j}(:,1)~=sort(data{1,j}(:,1)))>0
                       s = data{1,j}(:,1);
                       fr = data{1,j}(:,2);
                       A(n,:) = lsqcurvefit(tuning_curve, A0, s, fr, lb);
                       n = n+1;
                       save(['single_tuning_curves/tuning_' num2str(j), '_' , file(k).name], 's', 'fr');
               end
        end
end

r = A(:,1);
R = A(:,2);
theta = A(:,3);
sigma = A(:,4);
norm_r = r./R;
