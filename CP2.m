%%% Problem 1

%%% Initialize A as a 20 by 21 matrix of zeros (Week 2 Lecture 1)
%%% To fill in the matrix create a nested for loop (Week 2 Lecture 1)
%%% Save the matrix A as the variable A1

A = zeros(20,21);

for i = 1:20
    for j = 1:21
        A(i,j) = 1/(i*j);
    end
end

A1 = A;

%%% Let B equal A, and set the entire 15th row as zero (Week 1 Lecture 3)
%%% Do the same thing for the entire 16th column
%%% Save the matrix B as A2

B = A;

B(15,:) = 0;
B(:,16) = 0;
A2 = B;

%%% For A3, since we want the last few columns/rows you want to use the end
%%% command (Week 1 Lecture 3)

A3 = B(18:20, 17:21);


%%% Set A4 as the 10th column of B (Week 1 Lecture 3)


A4 = B(:,10);

%%% Problem 2

%%% For A5 and A6 it's exactly like Week 2 Theory lecture.
A5 = 0;
for i = 1:20
    A5 = A5 + 1/i;
end

A6 = 0;
for i = 1:200
    A6 = A6 + 1/i;
end

%%% For A7 through A10 you're still doing a Sum as you did for A5 and A6
%%% but now you want to break out of the loop when the sum surpasses 10
%%% for A7 and A8, and 20 for A9 and A10
%%% (very similar to Week 2 Lecture 2 Fibonacci)

A7 = 1;
A8 = 1;

while A8 < 10
    A7 = A7 + 1;
    A8 = A8 + 1/A7;
end

A9 = 1;
A10 = 1;

while A10 < 20
    A9 = A9 + 1;
    A10 = A10 + 1/A9;
end

%%% Problem 3
%%% First go to the bottom of this document to create a function because in
%%% MATLAB functions go at the end of the m-file, then come back here.

%%% After you have made a function at the end of the m-file you can start
%%% assigning your variables here.

%%% Set N and x0 according to the assignment file

N = 100;
x0 = 0.2;


%%% For each of the next three set its respective r value.  Then set the
%%% vector x as the iterates of the logistic map using the output of the
%%% function you created.

%%% Write the code for A11 and A12 here
r = 2.75;

A11 = zeros(1,100);
A11(1) = x0;
for i = 1:99
    A11(i+1) = logistic(x0, r, i);
end

behavior = 1;

x_std = std(A11);

A12 = [x_std, behavior];

%%% Do the same as above except for A13 and A14
r = 3.25;

A13 = zeros(1,100);
A13(1) = x0;
for i = 1:99
    A13(i+1) = logistic(x0, r, i);
end

behavior = 2;

x_std = std(A13);

A14 = [x_std, behavior];


%%% Do the same as above except for A15 and A16
r = 3.75;

A15 = zeros(1,100);
A15(1) = x0;
for i = 1:99
    A15(i+1) = logistic(x0, r, i);
end

behavior = 3;

x_std = std(A15);

A16 = [x_std, behavior];


%%% Create a function here (Week 2 Lecture 3).  The function will take r, 
%%% x0, and N as inputs and output a vector x of all N iterates starting at
%%% x0.  Inside the function create a loop that calculates the value of the
%%% logistic map and saves it in its respective entries of x.
function x = logistic(x0, r, N)
    x = x0*r*(1-x0);
    if N > 1
        for i = 2:N
            x = x*r*(1-x);
        end
    end
end

