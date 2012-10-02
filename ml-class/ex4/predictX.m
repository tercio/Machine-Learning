function predictX (X,Theta1,Theta2),

	displayData (X,20);
	printf ('O numero lido é: %d\n',mod (double (predict (Theta1,Theta2,X)),10)  );

end;
