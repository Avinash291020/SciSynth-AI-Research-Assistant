% symbolic_rules.pl

% If X causes Y, and Y causes Z, then X indirectly affects Z.
causes(x, y).
causes(y, z).
indirect(X, Z) :- causes(X, Y), causes(Y, Z). 