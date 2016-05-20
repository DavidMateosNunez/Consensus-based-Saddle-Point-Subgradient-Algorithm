function rate=learnrate_doubling_trick(itemax)
% Computes a sequence of length itemax of learning rates according to the 
% Doubling Trick


rate=zeros(1,itemax-1);

for m = 0:ceil(log2(itemax))
   
    for t=2^m :(2^(m+1)-1)
    
     rate(t) = 1/(2^m)^(1/2);   
    end
end