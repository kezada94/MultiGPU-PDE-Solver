#include <fstream>
#include <iostream>
#include <ginac/ginac.h>
using namespace std;
using namespace GiNaC;


const int dots = 3;
symbol ndmat[dots][dots][dots][dots][dots];
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

DECLARE_FUNCTION_2P(deldel)
DECLARE_FUNCTION_2P(delf)
DECLARE_FUNCTION_1P(box)

REGISTER_FUNCTION(a, latex_name("\\alpha"))
REGISTER_FUNCTION(f, latex_name("F"))
REGISTER_FUNCTION(g, latex_name("G"))
REGISTER_FUNCTION(deldel, latex_name("\\nabla_1"))
REGISTER_FUNCTION(delf, latex_name("\\nabla_2"))
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
	ex f = notation.op(0);
	ex g = notation.op(1);
	ex res = -f.diff(t)*g.diff(t) + l1*f.diff(r)*g.diff(r) + l1*f.diff(theta)*g.diff(theta) + l2*f.diff(phi)*g.diff(phi);
	return res;
}

ex translateDelf(const ex &notation){
	ex g = notation.op(0);
	ex f = notation.op(1);
	ex res = -(g*f.diff(t)).diff(t) + l1*(g*f.diff(r)).diff(r) + l1*(g*f.diff(theta)).diff(theta) + l2*(g*f.diff(phi)).diff(phi);
	return res;
}

ex replace(const ex &e, const ex &notation, auto (*func)(auto)){
	ex result = subs(e, notation == func(notation));
	//ex canon = -p.diff(t)*q.diff(t) + l1*p.diff(r)*q.diff(r) + l1*p.diff(theta)*q.diff(theta) + l2*p.diff(phi)*q.diff(phi);
	return result;
}

ex replaceAll(const ex &e, const ex &canon, auto (*func)(auto)){
	ex res = e;
	exset matches = getMatches(e, canon);
	cout << matches << endl;
	for(auto it = matches.begin(); it != matches.end(); ++it){
		ex ans = *it;
		res = replace(res, ans, func);
	}
	return res;
}

ex replaceAllBox(const ex &e){
	ex canon = box(p);
	ex res = replaceAll(e, canon, &translateBox);
	canon = box(p)*s;
	res = replaceAll(res, canon, &translateBox);
	return res;
}

ex replaceAllDelDel(const ex &e){
	ex canon = deldel(p, q);
	ex res = replaceAll(e, canon, &translateDelDel);
	canon = deldel(p, q)*s;
	res = replaceAll(res, canon, &translateDelDel);

	return res;
}

ex replaceAllDelf(const ex &e){
	ex canon = delf(p, q);
	ex res = replaceAll(e, canon, &translateDelf);
	canon = delf(p, q)*s;
	res = replaceAll(res, canon, &translateDelf);
	//canon = n3(p)*n4(q);
	//res = replaceAll(res, canon, &replaceDelDel);
	return res;
}

symbol getSymbolFromID(unsigned int id){
	if (id == 0){
		return t;
	} else if (id == 1) {
		return r;
	} else if (id == 2) {
		return theta;
	} else if (id == 3) {
		return phi;
	} 
	throw "Error. variable ID not implemented.";
}

symbol getDelta(int i){
	switch(i){
		case 0:
			return dt;
			break;
		case 1:
			return dr;
			break;
		case 2:
			return dtheta;
			break;
		case 3:
			return dphi;
			break;
	}
	throw "not implemented";
}

unsigned int getFuncIndex(const ex &e){
	if (e.match(a(t, r, theta, phi))){
		return 0;
	}
	if (e.match(f(t, r, theta, phi))){
		return 1;
	}
	if (e.match(g(t, r, theta, phi))){
		return 2;
	}
	throw "Not implemented.";
}
int isThis(unsigned int id, auto map){
	if (map.size() != 0 && map.find(id) != map.end()){
		return map[id];
	} else {
		return 0;
	}
}

symbol getDiscreteSymbol(const ex &f, auto map){
	return ndmat[getFuncIndex(f)][1+isThis(0, map)][1+isThis(1, map)][1+isThis(2, map)][1+isThis(3, map)];
}

ex translateToFOD(const ex &e, auto ms){
	unsigned int id = *ms.begin();
	symbol v = getSymbolFromID(id);
	map<unsigned int, int> map1 = {{id , 1}};
	map<unsigned int, int> map2 = {{id , -1}};
    ex res = (getDiscreteSymbol(e, map1) - getDiscreteSymbol(e, map2))/ (2*getDelta(id));

	return res;
}

ex translateToSOD(const ex &e, auto ms){
	unsigned int id1 = *ms.begin();
	unsigned int id2 = *(++ms.begin());
	symbol v1 = getSymbolFromID(id1);
	symbol v2 = getSymbolFromID(id2);
	ex res;
	if (id1 == id2){
		map<unsigned int, int> map1 = {{id1, 1}};
		map<unsigned int, int> map2 = {{id1, -1}};
		map<unsigned int, int> map3;
    	res = (getDiscreteSymbol(e, map1) - 2*getDiscreteSymbol(e, map3) + getDiscreteSymbol(e, map2))/ (pow(getDelta(id1), 2));
	} else {
		map<unsigned int, int> map1 = {{id1, 1}, {id2, 1}};
		map<unsigned int, int> map2 = {{id1, -1}, {id2, 1}};
		map<unsigned int, int> map3 = {{id1, 1}, {id2, -1}};
		map<unsigned int, int> map4 = {{id1, -1}, {id2, -1}};
		res = (getDiscreteSymbol(e, map1) - getDiscreteSymbol(e, map2) - getDiscreteSymbol(e, map3) + getDiscreteSymbol(e, map4))/ (getDelta(id1)*getDelta(id2));
	}
	return res;
}

ex replaceAllDerivatives(const ex &e){
	ex res = e;
	cout << "Segment: " << e << endl;
	stack<ex> stack;
	stack.push(e);

    while (!stack.empty()) {
 
        ex curr = stack.top();
        stack.pop();
 
		if (is_a<fderivative>(curr)){
			fderivative der = ex_to<fderivative>(curr);
			GiNaC::function fun = ex_to<GiNaC::function>(curr);
			cout << "This is the info: as derivative = "<< der << " and as function= " << fun << endl; 
			auto ms = der.derivatives();
			ex disc;
			if (ms.size() == 1){
				disc = translateToFOD(fun, ms);
			} else if (ms.size() == 2){
				disc = translateToSOD(fun, ms);
			} else {
				throw "Error. Derivative order not implemented.";
			}
			res = res.subs(curr == disc);
			//for(auto it = ms.begin(); it != ms.end(); ++it){
			//	auto ans = *it;
			//}
		}
		if(curr.nops() != 0) {
			for (int i=0; i<curr.nops(); i++){
				stack.push(curr.op(i));
			}
		}
    }
	return res;
}
bool isCanon(const ex &e){
	if (e.match(f(t,r,theta,phi)) || e.match(a(t,r,theta,phi)) || e.match(g(t,r,theta,phi))){
		return true;
	}
	return false;
}

ex replaceAllFunctionsDiscrete(const ex &e){
	ex res = e;
	stack<ex> stack;
	stack.push(e);

    while (!stack.empty()) {
 
        ex curr = stack.top();
        stack.pop();
 
		if (isCanon(curr)){
			map<unsigned int, int> mapa;
			res = res.subs(curr == getDiscreteSymbol(curr, mapa));
		}
		if(curr.nops() != 0) {
			for (int i=0; i<curr.nops(); i++){
				stack.push(curr.op(i));
			}
		}
    }
	return res;
}
string getFuncName(int i){
    if (i==0)
        return "\\alpha";
    if (i==1)
        return "F";
    if (i==2)
        return "G";
	throw "Not implemented.";
}

string i2d(int i){
    if (i==0)
        return "-1";
    if (i==1)
        return "";
    if (i==2)
        return "+1";
	throw "Not implemented.";
}

int main2() {
	ex alfa = a(t, r, theta, phi);
	ex F = f(t, r, theta, phi);
	ex G = g(t, r, theta, phi);
	
	for (int _f=0; _f<dots; _f++){
		for (int _t=0; _t<dots; _t++){
			for (int _r=0; _r<dots; _r++){
				for (int _theta=0; _theta<dots; _theta++){
					for (int _phi=0; _phi<dots; _phi++){
						ndmat[_f][_t][_r][_theta][_phi] = symbol(getFuncName(_f)+"^{n"+i2d(_t)+"}_{i"+i2d(_r)+"j"+i2d(_theta)+"k"+i2d(_phi)+"}" );
					}
				}
			}
		}
	}
	cout << ndmat[0][1][1][1][1]<< endl;
	//ex test = (-box(alfa) + sin(alfa)*cos(alfa)*(delta*efe*delta*efe + sin(efe)*sin(efe)*delta*ge*delta*ge));
	ex test = box(F);
	ex test2 = deldel(F, alfa);
	ex test3 = delf(sin(F)*G, F);
	ex test4 = (box(alfa) + sin(alfa)*cos(alfa)*(deldel(F, F) + pow(sin(F), 2)*deldel(G, G))) + 
		l*(sin(alfa)*cos(alfa)*( (deldel(alfa, alfa)) * (deldel(F, F)) - pow(deldel(alfa, F), 2)) +
         	sin(alfa)*cos(alfa)*pow(sin(F), 2)*((deldel(alfa, alfa))*(deldel(G, G)) - pow(deldel(alfa, G), 2)) + 
         	2*pow(sin(alfa), 3)*cos(alfa)*pow(sin(F), 2)*((deldel(F, F))*(deldel(G, G)) - pow(deldel(F, G), 2)) -
         delf(pow(sin(alfa), 2)*(deldel(F, F)), alfa) + delf(pow(sin(alfa), 2)*(deldel(alfa, F)), F) -
         delf(pow(sin(alfa), 2)*pow(sin(F), 2)*(deldel(G, G)), alfa) + delf(pow(sin(alfa), 2)*pow(sin(F), 2)*(deldel(alfa, G)), G) );


	cout << endl <<endl;
	cout << "Replace Box now: " << endl;
	ex test5 = replaceAllBox(test4);
	cout << "Replace DELDEL now: " << endl;
	ex test6 = replaceAllDelDel(test5);
	cout << "Replace Delf now: " << endl;
	ex test7 = replaceAllDelf(test6);
	

	cout << "Replace FOD now: " << endl;
	test7 = replaceAllDerivatives(test7);

	cout << "Replace FOD now: " << endl;
	test7 = replaceAllFunctionsDiscrete(test7);

	cout << "FODnow: " << endl;
	cout << is_a<fderivative>(F.diff(t)) << endl;
	F.diff(t).dbgprinttree();

	cout << "SODnow: " << endl;
	cout << is_a<fderivative>(F.diff(t).diff(t)) << endl;
	F.diff(t).diff(r).dbgprinttree();
	fderivative kepa = ex_to<fderivative>(F.diff(t).diff(r));
	GiNaC::function kepa2 = ex_to<GiNaC::function>(F.diff(t).diff(r));
	cout << kepa2 << endl;
	auto ms = kepa.derivatives();
	exvector ev; 
	ex fun = kepa.thiscontainer(ev);
	cout << fun << endl;
	exmap ekke;
	cout << kepa2.match(f(t,r,theta,phi), ekke) << endl;

	

	cout << "Container: "<< endl;
	for(auto gh : ev){
		cout << gh << endl;
	}
	cout << "Params " << ms.size()<< endl;
	for(auto it = ms.begin(); it != ms.end(); ++it){
		auto ans = *it;
		cout << ans << endl;
	}







	std::ofstream file;
	print_latex pl1(cout);
	print_latex pl2(file);
	file.open("text.txt");
	test7.print(pl1);
	test7.print(pl2);
	file.close();
	cout << endl;
	



		
	//ex resu = lsolve(test7.expand()==0, ndmat[0][1][1][1][1]);
	exmap as;
	ex resu = factor(test7);
	resu.print(pl1);
	return 0;
}


