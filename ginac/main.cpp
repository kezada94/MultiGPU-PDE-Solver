#include <fstream>
#include <iostream>
#include <ginac/ginac.h>
using namespace std;
using namespace GiNaC;


symbol dt("\\Delta{t}"), dr("\\Delta{r}"), dtheta("\\Delta{\\theta}"), dphi("\\Delta{\\Phi}");
symbol l1("l1"), l2("l2"), l3("l3");
symbol l("\\lambda");
symbol t("t"), r("r"), theta("\\theta"), phi("\\Phi");
/*
ex del(const ex &f){
	ex canon = -f.diff(v[0], 1) + l1*f.diff(v[1], 1)+ l2*f.diff(v[2], 1)+ l3*f.diff(v[3], 1);
	return canon;
}

ex deldel(const ex &f, const ex &g){
	ex canon = -f.diff(v[0], 1)*g.diff(v[0], 1) + l1*f.diff(v[1], 1)*g.diff(v[1], 1) + l2*f.diff(v[2], 1)*g.diff(v[2], 1) + l3*f.diff(v[3], 1)*g.diff(v[3], 1);
	return canon;
}

ex box(const ex &f){
	ex canon = -f.diff(v[0], 2) + l1*f.diff(v[1], 2)+ l2*f.diff(v[2], 2)+ l3*f.diff(v[3], 2);
	return canon;
}
*/
int main2();

int main() {
	main2();
	return 0;
}


DECLARE_FUNCTION_4P(a)
DECLARE_FUNCTION_4P(f)
DECLARE_FUNCTION_4P(g)

DECLARE_FUNCTION_1P(n1)
DECLARE_FUNCTION_1P(n2)
DECLARE_FUNCTION_1P(n3)
DECLARE_FUNCTION_1P(n4)
DECLARE_FUNCTION_1P(box)

REGISTER_FUNCTION(a, latex_name("\\alpha"))
REGISTER_FUNCTION(f, latex_name("F"))
REGISTER_FUNCTION(g, latex_name("G"))
REGISTER_FUNCTION(n1, latex_name("\\nabla_\\mu"))
REGISTER_FUNCTION(n2, latex_name("\\nabla^\\mu"))
REGISTER_FUNCTION(n3, latex_name("\\nabla_\\nu"))
REGISTER_FUNCTION(n4, latex_name("\\nabla^\\nu"))
REGISTER_FUNCTION(box, latex_name("\\Box"))

ex p = wild(1);
ex q = wild(2);	
ex s = wild(3);	

exset getMatches(const ex &e, const ex &canon){
	exset se = exset();
	e.find(canon, se);
	return se;
}

ex translateBox(const ex &notation){
	ex f = notation.op(0);
	ex res = -f.diff(t, 2) + l1*(f.diff(r,2) + f.diff(theta, 2)) + l2*(f.diff(phi, 2));
	return res;
}

ex translateDelDel(const ex &notation){
	ex f = notation.op(0).op(0);
	ex g = notation.op(1).op(0);
	ex res = -f.diff(t)*g.diff(t) + l1*f.diff(r)*g.diff(r) + l1*f.diff(theta)*g.diff(theta) + l2*f.diff(phi)*g.diff(phi);
	return res;
}

ex translateDelf(const ex &notation){
	exmap aaa = exmap();
	notation.match(n1(q*n2(p)), aaa);
	ex f = aaa[p];
	ex g = aaa[q];
	ex res = -(g*f.diff(t)).diff(t) + l1*(g*f.diff(r)).diff(r) + l1*(g*f.diff(theta)).diff(theta) + l2*(g*f.diff(phi)).diff(phi);
	return res;
}

ex replace(const ex &e, const ex &notation, auto (*func)(auto)){
	ex result = subs(e, notation == func(notation));
	//ex canon = -p.diff(t)*q.diff(t) + l1*p.diff(r)*q.diff(r) + l1*p.diff(theta)*q.diff(theta) + l2*p.diff(phi)*q.diff(phi);
	return result;
}

ex replaceBox(const ex &e, const ex &term) {
	ex result = replace(e, term, &translateBox);
	return result;
}

ex replaceDelDel(const ex &e, const ex &term) {
	ex result = replace(e, term, &translateDelDel);
	return result;
}

ex replaceDelf(const ex &e, const ex &term) {
	ex result = replace(e, term, &translateDelf);
	return result;
}

ex replaceAll(const ex &e, const ex &canon, auto (*func)(auto, auto)){
	ex res = e;
	exset matches = getMatches(e, canon);
	cout << matches << endl;
	for(auto it = matches.begin(); it != matches.end(); ++it){
		ex ans = *it;
		res = func(res, ans);
	}
	return res;
}

ex replaceAllBox(const ex &e){
	ex canon = box(p);
	ex res = replaceAll(e, canon, &replaceBox);
	canon = box(p)*s;
	res = replaceAll(res, canon, &replaceBox);
	return res;
}

ex replaceAllDelDel(const ex &e){
	ex canon = n3(p)*n4(q);
	ex res = replaceAll(e, canon, &replaceDelDel);
	canon = n3(p)*n4(q)*s;
	res = replaceAll(res, canon, &replaceDelDel);

	canon = n1(p)*n2(q);
	res = replaceAll(res, canon, &replaceDelDel);
	canon = n1(p)*n2(q)*s;
	res = replaceAll(res, canon, &replaceDelDel);
	return res;
}

ex replaceAllDelf(const ex &e){
	ex canon = n1(p*n2(q));
	ex res = replaceAll(e, canon, &replaceDelf);
	canon = n1(p*n2(q))*s;
	res = replaceAll(res, canon, &replaceDelf);
	//canon = n3(p)*n4(q);
	//res = replaceAll(res, canon, &replaceDelDel);
	return res;
}
int main2()
{
	
	ex alfa = a(t, r, theta, phi);
	ex F = f(t, r, theta, phi);
	ex G = g(t, r, theta, phi);
	
	//ex test = (-box(alfa) + sin(alfa)*cos(alfa)*(delta*efe*delta*efe + sin(efe)*sin(efe)*delta*ge*delta*ge));
	ex test = box(F)-box(alfa) ;
	ex test2 = n1(F)*n2(alfa) *F;
	ex test3 = n1(sin(F)*G*n2(F));
	ex test4 = (box(alfa) + sin(alfa)*cos(alfa)*(n1(F)*n2(F) + pow(sin(F), 2)*n1(G)*n2(G))) + 
		l*(sin(alfa)*cos(alfa)*( (n1(alfa)*n2(alfa)) * (n3(F)*n4(F)) - pow(n1(alfa)*n2(F), 2)) +
         	sin(alfa)*cos(alfa)*pow(sin(F), 2)*((n1(alfa)*n2(alfa))*(n3(G)*n4(G)) - pow(n1(alfa)*n2(G), 2)) + 
         	2*pow(sin(alfa), 3)*cos(alfa)*pow(sin(F), 2)*((n1(F)*n2(F))*(n3(G)*n4(G)) - pow(n1(F)*n2(G), 2)) -
         n1(pow(sin(alfa), 2)*(n3(F)*n4(F))*n2(alfa))+n1(pow(sin(alfa), 2)*(n3(alfa)*n4(F))*n2(F)) -
         n1(pow(sin(alfa), 2)*pow(sin(F), 2)*(n3(G)*n4(G))*n2(alfa))+n1(pow(sin(alfa), 2)*pow(sin(F), 2)*(n3(alfa)*n4(G))*n2(G)) );


	cout << endl <<endl;
	//cout << test.args << endl;
	ex test5 = replaceAllBox(test4);
	ex test6 = replaceAllDelDel(test5);
	ex test7 = replaceAllDelf(test6);
	

	std::ofstream file;
	print_latex pl1(cout);
	print_latex pl2(file);
	file.open("text.txt");
	test7.print(pl1);
	test7.print(pl2);
	file.close();
	cout << endl;
	return 0;
}


