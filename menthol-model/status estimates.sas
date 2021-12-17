

proc datasets nolist library=work kill;
run;
quit;

proc import datafile="Z:\chris\consulting\sarah mills\markov models\data\Beta Estimates 234.xlsx"
             out=_beta2_ dbms=xlsx;
run;
proc import datafile="Z:\chris\consulting\sarah mills\markov models\data\Beta Estimates 15.xlsx"
             out=_beta1_ dbms=xlsx;
run;
proc datasets nolist;
  contents data=_beta2_ order=varnum;
run;
quit;
proc print data=_beta2_;
run;



/* _BETA1_ */
/* 
/*
state21=0 if state2=5
state21=1 if state2=1

state31=0 if state3=5
state31=1 if state3=1

initialage1=1 if initialage=1 (<18)
initialage2=1 if initialage=2 (>=18)
reference (non-smoker)
*/
/* betas for set 2 */
proc print data=_beta2_;
run;
data _b2_;
  set _beta2_;
  drop response _name_;
run;
proc transpose data=_b2_ out=_b2_(drop=_name_ _label_);
run;
/* simulated data */
data _obs2_;
  drop s2 s3 ia;
  one=1;
  do s2=1 to 4;
    do s3=2 to 4;
	  do ia=1 to 3;
	    do black=0 to 1;
		  do gen=0 to 1;
		    do pov=0 to 1;
			  age=30;
			  s21=(s2=1);
			  s22=(s2=2);
			  s23=(s2=3);
			  s24=(s2=4);
			  s32=(s3=2);
			  s33=(s3=3);
			  s34=(s3=4);
			  ia1=(ia=1);
			  ia2=(ia=2);
			  output;
			end;
		  end;
		end;
	  end;
	end;
  end;
run;
data _obs2_;
  retain one s21 s22 s23 s24 s32 s33 s34 ia1 ia2 black age gen pov;
  set _obs2_;
run;
proc iml;
  use _obs2_;
  read all into x;
  close _obs2_;
  use _b2_;
  read all into b;
  close _b2_;
  xb=x*b;
  create _results_ from xb;
  append from xb;
quit;
/* back transform */
data _results_;
  set _results_;
  r2_r4=exp(col1);
  r3_r4=exp(col2);
  r5_r4=exp(col3);
  p4=1/(1+r2_r4+r3_r4+r5_r4);
  p2=p4*r2_r4;
  p3=p4*r3_r4;
  p5=p4*r5_r4;
  foo=p4+p2+p3+p5;
run;


proc print data=_results_(obs=10);
run;



		   
