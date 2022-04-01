clc;
close all;

X_train = matfile('X_train_lbp.mat');
X_train = X_train.X_train;
y_train = matfile('y_train_lbp.mat');
y_train = y_train.y_train;

X_test = matfile('X_test_lbp.mat');
X_test = X_test.X_test;
y_test = matfile('y_test_lbp.mat');
y_test = y_test.y_test;

test_size = size(X_test);
test_size = test_size(1);

%svm classifier
clf = fitcecoc(X_train,y_train,'coding','binarycomplete');
pred = predict(clf,X_test);

accuracy = (sum(pred == y_test)/test_size)*100;
display(accuracy);

C = confusionmat(y_test,pred); %confusion matrix
confusionchart(C);

%saving the model
save('clf.mat','clf');