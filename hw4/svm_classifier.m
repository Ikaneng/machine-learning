function YTest=svm_classifier(kernel,method,XTest,XTrain,YTrain,showPlot)
    % SVM_CLASSIFIER 
    % Returns a label Ytest to the test using svmtrain and svmclassify
    Mdl=svmtrain(XTrain,YTrain,'ShowPlot',showPlot,'kernel_function',kernel,'method',method);
    YTest=svmclassify(Mdl,XTest);
end