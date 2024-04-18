import semopy

model = '''
f1 BY pc12 pc13 pc14 pc15 pc16 pc17@1;
f2 BY pc23 pc24 pc25 pc26 pc27@1;
f3 BY pc34 pc35 pc36 pc37@1;
f4 BY pc45 pc46 pc47@1;
f5 BY pc56 pc57@1;
f6 BY pc67@1;

pc12-pc67*.1;

f1@1;
f7@1;

f2 WITH f1;
f3 WITH f1 f2;
f4 WITH f1 f3;
f5 WITH f1 f4;
f6 WITH f1 f5;

f7 WITH f1 f6@0;

pc12-pc67*.1;

OUTPUT: TECH1; TECH5;
'''

model_fit = semopy.Model(model)
model_fit.fit('cars pc.dat')
model_fit.inspect()
