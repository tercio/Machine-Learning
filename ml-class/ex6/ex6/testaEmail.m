function p = testaEmail (filename,model),

	% Read and predict
	file_contents = readFile(filename);
	word_indices  = processEmail(file_contents);
	x             = emailFeatures(word_indices);
	p = svmPredict(model, x);

end;


