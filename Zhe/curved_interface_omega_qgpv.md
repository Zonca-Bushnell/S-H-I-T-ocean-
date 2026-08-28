# A-only 鏇查潰/绛夊瘑搴﹂潰 QGPV锛氫粠鏍囧噯灞傛ā鍨嬪埌 revised curved-surface QGPV

## 鎽樿

鏈枃灏嗗師鏉ョ殑澶氬垎鏀粡楠屼慨姝ｄ綋绯讳慨璁负 **baseline / revised A** 涓ょ被妯″瀷銆傛湰鏂囧彧璁ㄨ鏍囧噯 QGPV 涓?revised curved-surface QGPV锛涙棫鐨勯潪鍦拌浆璇婃柇鍒嗘敮涓嶅啀浣滀负姝ｅ紡鐞嗚妯″瀷鍙備笌鏈枃鎺ㄥ銆侀獙璇佹垨鍥捐〃瑙ｉ噴銆?

鏈枃鐨?revised A model 涓嶆槸鍗曠嫭鍙犲姞涓€涓粡楠屼慨姝ｉ」锛岃€屾槸鎶婃爣鍑?QGPV 鐨勬按骞冲钩闈㈢畻瀛愭浛鎹负绛夊瘑搴﹂潰涓婄殑鏇查潰绠楀瓙銆傜悊璁轰富绾夸负锛?

```math
\text{standard QGPV}
\rightarrow
S_i:\rho=\rho_i
\rightarrow
g_{ab},\ g^{ab},\ \Delta_g
\rightarrow
q_i^{A}
\rightarrow
\text{tilted eddy response}.
```

鏍稿績鎬濇兂鏄細绛夊瘑闈?\(S_i\) 鏄墿璐ㄩ潰锛屾丁鏃嬪€炬枩棣栧厛琛ㄧ幇涓轰笉鍚屽瘑搴﹂潰涓婄殑娴佸嚱鏁般€佹丁蹇冨拰绛夊瘑闈㈠嚑浣曞彂鐢熺浉瀵瑰亸绉汇€傚洜姝わ紝閫傚悎鎻忚堪璇ヨ繃绋嬬殑妯″瀷搴旂洿鎺ュ湪 \(S_i\) 鐨勫垏骞抽潰涓婂畾涔夊湴杞祦銆佹洸闈?Laplacian 鍜?QGPV锛岃€屼笉鏄厛鍦ㄥ浐瀹?\(p\) 鍧愭爣涓瘖鏂瘑搴﹂潰褰㈠彉銆?

## 0. 绗﹀彿鍜屽潗鏍囩害瀹?

璁句笁缁翠綅缃负 \(\mathbf x=(x,y,z)\)銆傚瘑搴︿负 \(\rho(\mathbf x,t)\)锛屽弬鑰冨瘑搴︿负 \(\rho_0\)銆傜 \(i\) 涓瓑瀵嗗害闈㈠畾涔変负

```math
S_i(t)=\{\mathbf x:\rho(\mathbf x,t)=\rho_i\}.
```

鍦?\(S_i\) 涓婂彇灞€鍦版洸闈㈠潗鏍?\(\xi^1,\xi^2\)锛屾洸闈㈠祵鍏ュ啓浣?

```math
\mathbf r_i(\xi^1,\xi^2,t)
=
\mathbf r_i^0(\xi^1,\xi^2)
+\eta_i(\xi^1,\xi^2,t)\mathbf n_i^0(\xi^1,\xi^2),
```

鍏朵腑 \(\eta_i\) 涓虹瓑瀵嗗害闈㈢殑娉曞悜浣嶇Щ锛孿(\mathbf n_i^0\) 涓哄熀鍑嗙瓑瀵嗗害闈㈢殑鍗曚綅娉曞悜閲忋€傛洸闈㈠垏鍚戝熀鐭负

```math
\mathbf r_a=\frac{\partial \mathbf r_i}{\partial \xi^a},
\qquad a=1,2.
```

鏇查潰搴﹂噺寮犻噺瀹氫箟涓?

```math
g_{ab}=\mathbf r_a\cdot\mathbf r_b,
\qquad
g=\det(g_{ab}),
\qquad
g^{ab}=(g_{ab})^{-1}.
```

褰撴洸闈㈠彲鍐欐垚鍥惧舰闈?\(z=Z_i(x,y,t)\) 鏃讹紝

```math
g_{xx}=1+Z_x^2,\qquad
g_{xy}=Z_xZ_y,\qquad
g_{yy}=1+Z_y^2,
```

骞朵笖

```math
g=1+Z_x^2+Z_y^2=1+|\nabla Z|^2.
```

鏇查潰 Laplace-Beltrami 绠楀瓙瀹氫箟涓?

```math
\Delta_g\psi
=
\frac{1}{\sqrt g}
\frac{\partial}{\partial \xi^a}
\left(
\sqrt g\,g^{ab}
\frac{\partial \psi}{\partial \xi^b}
\right).
```

閲嶅鎸囨爣 \(a,b\) 閲囩敤 Einstein 姹傚拰绾﹀畾銆傛祦鍑芥暟 \(\psi_i\) 鍦?\(S_i\) 涓婂畾涔夛紝鏇查潰鍦拌浆閫熷害涓?

```math
u_i^a
=
\frac{1}{f_0}\epsilon^{ab}\nabla_b\psi_i,
```

鍏朵腑 \(\epsilon^{ab}\) 鏄洸闈笂鐨勫弽瀵圭О寮犻噺銆傚湪灞€鍦板钩闈㈡瀬闄愪笅锛?

```math
u_i=-\frac{\partial\psi_i}{\partial y},
\qquad
v_i=\frac{\partial\psi_i}{\partial x}.
```

## 1. 鏍囧噯 QGPV 鐨勯棶棰?

鏍囧噯杩炵画 QGPV 鍙啓涓?

```math
q
=
f_0+\beta y+\nabla_h^2\psi
+
\frac{\partial}{\partial z}
\left(
\frac{f_0^2}{N^2}
\frac{\partial\psi}{\partial z}
\right).
```

杩欓噷 \(\nabla_h^2=\partial_x^2+\partial_y^2\) 鏄浐瀹氭按骞冲钩闈笂鐨?Laplacian锛孿(N^2\) 涓烘诞鍔涢鐜囧钩鏂广€傝褰㈠紡閫傚悎寮变綅绉汇€佸急鍊炬枩骞朵笖瀵嗗害闈㈡帴杩戞按骞崇殑鎯呭喌銆?

浣嗘槸鍦ㄥ€炬枩娑℃棆涓紝瀵嗗害闈㈡湰韬彂鐢熷彲瑙傚集鏇诧紝涓嶅悓灞傛丁蹇冧篃浼氫骇鐢熺浉瀵逛綅绉汇€傝嫢浠嶆妸鎵€鏈夋按骞冲鏁伴兘鏀惧湪鍥哄畾 \((x,y)\) 骞抽潰涓紝鍒欑瓑瀵嗛潰鍑犱綍鍙€氳繃鍚庡鐞嗚瘖鏂繘鍏ワ紝瀹规槗鎶婄墿璐ㄩ潰褰㈠彉瑙ｉ噴鎴愬浐瀹氬潗鏍囦笅鐨勫瀭鍚戞贩鍚堟垨铏氬亣 stretching銆?

鍥犳 revised A model 鐨勫嚭鍙戠偣鏄細鍦?\(S_i\) 涓婄洿鎺ュ畾涔?PV锛岃€屼笉鏄厛鍦?\(p\) 鍧愭爣鎴栧浐瀹氭繁搴﹀潗鏍囦腑鍐欏畬鏂圭▼鍐嶈瘖鏂?\(S_i\)銆?

## 2. 绛夊瘑搴﹂潰鏄墿璐ㄩ潰

绛夊瘑搴﹂潰鐨勬牳蹇冪害鏉熸槸

```math
\frac{D\rho}{Dt}=0.
```

鍥犳瀵?\(S_i:\rho=\rho_i\)锛屾湁

```math
\frac{D}{Dt}(\rho-\rho_i)=0.
```

鑻ョ敤娉曞悜浣嶇Щ \(\eta_i\) 鎻忚堪鏇查潰杩愬姩锛屽垯鏇查潰娉曞悜閫熷害涓?

```math
w_{n,i}
=
\frac{D\mathbf r_i}{Dt}\cdot\mathbf n_i^0
=
\partial_t\eta_i+u_i^a\nabla_a\eta_i.
```

杩欎釜寮忓瓙璇存槑锛氱瓑瀵嗛潰褰㈠彉涓嶆槸澶栧姞鍥惧舰锛岃€屾槸琚垏鍚戞祦鍜屾硶鍚戣繍鍔ㄥ叡鍚屾帹杩涚殑鐗╄川闈㈠嚑浣曘€傚€炬枩娑℃棆鐨勨€滃眰闂存丁蹇冨亸绉烩€濆簲鐞嗚В涓轰笉鍚?\(S_i\) 涓婃祦鍑芥暟缁撴瀯鐨勭浉瀵瑰钩绉诲拰鏇查潰鍑犱綍宸紓銆?

## 3. 琛ㄩ潰鑺傜偣涓庡唴閮ㄦ祦褰㈢殑缁熶竴 PV 鍙嶆紨

鍘熷 CMEMS 鏁版嵁鍖呭惈娴烽潰楂樺害 `zos_glor`锛屾湰宸ョ▼鍦ㄦ瘡鏃ヨ緭鍏ヤ腑璁颁负 `adt`锛屽湪鐢熷懡鍛ㄦ湡鍚堟垚涓涓?`adt_anom`銆傚洜姝ゆ捣琛ㄤ笉鑳戒綔涓哄垰鎬х洊锛屼篃涓嶅簲浣滀负涓庡唴閮ㄥ绔嬬浉鍔犵殑澶栭儴琛ヤ竵銆傛寜鐓?Bretherton 杈圭晫 PV 涓?surface-QG 鐨勬€濇兂锛屾捣琛ㄨ竟鐣屼笌鍐呴儴 PV 鏄悓涓€涓き鍦嗗弽婕旈棶棰樼殑杈圭晫婧愬拰浣撴簮銆傛湰鏂囨妸鑷敱闈㈠啓鎴愮 0 涓祦褰㈣妭鐐?
```math
S_0=S_\eta:\ z=\eta(x,y,t),
```

鍐呴儴绛夊瘑搴﹂潰浠嶄负

```math
S_i(t)=\{\mathbf x:\sigma_0(\mathbf x,t)=\rho_i\},\qquad i=1,\dots,N.
```

杩欓噷 \(S_0\) 涓?\(S_i\) 鍏卞悓鏋勬垚涓€涓彲鍙樺舰娴佸舰缃戠粶銆俽evised A 鐨勬牳蹇冨垱鏂颁粛鐒舵槸锛氭瘡涓?\(S_i\) 閮藉彲浠ュ彂鐢熷嚬鍑搞€佸€炬枩鍜岀浉瀵瑰亸绉伙紝鑰屼笉鏄垰鎬у钩闈㈠钩绉汇€?
娴疯〃鍘嬪姏寮傚父涓?
```math
p_s'=\rho_0g\eta,
\qquad
\psi_\eta=g\eta.
```

鍥犳琛ㄩ潰鑺傜偣鐨勫湴杞€熷害涓?
```math
u_{g,\eta}=-\frac{g}{f_0}\frac{\partial\eta}{\partial y},
\qquad
v_{g,\eta}=\frac{g}{f_0}\frac{\partial\eta}{\partial x}.
```

鍦ㄥ悎鎴愬潗鏍?\(x^\ast=x/R,\ y^\ast=y/R\) 涓紝

```math
\psi_\eta^\ast=\frac{g\eta}{UR},
\qquad
u_{g,\eta}=-\frac{g}{f_0R}\frac{\partial\eta}{\partial y^\ast},
\qquad
v_{g,\eta}=\frac{g}{f_0R}\frac{\partial\eta}{\partial x^\ast}.
```

鑷敱闈㈠搴旂殑澶栨ā鍙樺舰鍗婂緞浠嶅畾涔変负

```math
R_{\eta,1}=\frac{\sqrt{gH_1}}{f_0},
```

浣嗗畠鍙彁渚涜〃闈㈣妭鐐逛笌椤跺眰鍐呴儴鑺傜偣涔嬮棿鐨勮€﹀悎寮哄害锛岃€屼笉鏄竴涓嫭绔嬪彔鍔犻」銆?
## 4. Morel 2019 涓庣瓑瀵嗛潰 PV 闂悎绾︽潫

Morel, Gula and Ponte (2019) 鎻愪緵浜嗕竴涓 revised A 鐗瑰埆閲嶈鐨勭害鏉燂細PV 涓嶅簲鍙鐪嬩綔鍥哄畾娣卞害缃戞牸涓婄殑灞€鍦扮偣鍊硷紝鑰屽簲鍚屾椂婊¤冻浣撶Н鍒嗕笌杈圭晫閫氶噺涔嬮棿鐨勯棴鍚堝叧绯汇€傚浠绘剰瀵嗗害鍦?\(\sigma_0(\mathbf x,t)\)锛孍rtel PV 鍙啓鎴愭暎搴﹀舰寮?
```math
PV
=
\nabla\cdot
\left(
\mathbf U_a\times\nabla\sigma_0
\right),
```

鍏朵腑 \(\mathbf U_a\) 涓虹粷瀵归€熷害锛孿(\nabla\sigma_0\) 涓烘€讳綅鍔垮瘑搴︽搴︺€傝繖涓舰寮忓彧渚濊禆鐭㈤噺寰Н鍒嗘亽绛夊紡锛屽洜姝ゅぉ鐒堕€傚悎涓€鑸洸闈㈠潗鏍囥€佺瓑瀵嗗害鍧愭爣鍜屾湰鏂囩殑鍙彉褰㈡祦褰?\(S_i:\sigma_0=\rho_i\)銆?
瀵逛换鎰忕敱涓ゅ紶鐩搁偦绛夊瘑闈㈠洿鎴愮殑鎺у埗浣?
```math
V_i(t)
=
\{
\mathbf x:
\rho_i\le \sigma_0(\mathbf x,t)\le\rho_{i+1}
\},
```

鍏惰竟鐣?\(\partial V_i\) 鍖呮嫭涓娿€佷笅绛夊瘑闈€佽〃闈㈣妭鐐?\(S_\eta\) 鍙兘浜х敓鐨?outcropping 杈圭晫銆佷晶杈圭晫浠ュ強鍙兘鐨勫簳杈圭晫銆傜敱鏁ｅ害瀹氱悊鍙緱

```math
\int_{V_i} PV\,dV
=
\oint_{\partial V_i}
\left(
\mathbf U_a\times\nabla\sigma_0
\right)\cdot d\mathbf S.
```

杩欎釜绛夊紡璇存槑锛歳evised A 鐨勬洸闈㈠嚑浣曢」涓嶈兘鍙湅灞€鍦?\(\Delta_g\psi\) 鐨勫舰鐘讹紝涔熷繀椤绘帴鍙?\(V_i\) 涓婄殑 PV 浣撶Н鍒嗕笌杈圭晫閫氶噺闂悎绾︽潫銆傝嫢涓€涓洸闈慨姝ｉ」鍦ㄥ眬閮ㄧ湅璧锋潵寰堝己锛屼絾鐮村潖浜嗚闂悎鍏崇郴锛屽垯瀹冨彧鑳戒綔涓?diagnostics锛屼笉鑳界洿鎺ヨ繘鍏ユ寮?QGPV forcing銆?
涓轰簡鎶婅繖涓害鏉熷啓鎴愬彲楠岃瘉閲忥紝瀹氫箟绗?\(i\) 涓瓑瀵嗛潰鎺у埗浣撳唴鐨?PV anomaly 涓?
```math
PVA_i
=
PV_i-PV_{i,\mathrm{ref}},
```

鍏朵腑 \(PV_{i,\mathrm{ref}}\) 鏄悓涓€瀵嗗害灞傚湪鍙傝€冮潤姝㈠眰缁撲笅鐨?PV銆傚搴旂殑闂悎娈嬪樊瀹氫箟涓?
```math
\epsilon_i^{PV}
=
\frac{
\left|
\int_{V_i} PVA_i\,dV
-
\oint_{\partial V_i}
\left(
\mathbf U_a\times\nabla\sigma_0
\right)\cdot d\mathbf S
\right|
}{
\left|\int_{V_i} PVA_i\,dV\right|+\epsilon
},
```

鍏朵腑 \(\epsilon\) 鏄伩鍏嶅垎姣嶄负闆剁殑灏忛噺銆傚悗缁暟鍊奸獙璇佷腑锛孿(\epsilon_i^{PV}\) 搴斾綔涓?A model 鏄惁鍙繘鍏ユ寮?forcing 鐨勮川閲忔帶鍒舵寚鏍囥€?
Morel 2019 瀵瑰绔嬫丁鏃嬭繕鎸囧嚭锛屽眰鍐?PVA 浣撶Н鍒嗕笌琛ㄩ潰瀵嗗害寮傚父銆佽〃闈㈡丁搴﹀強杈圭晫鏉′欢涔嬮棿瀛樺湪绾︽潫銆傚鏈枃鑰岃█锛岃繖鎰忓懗鐫€ \(S_\eta\) 涓嶆槸澶栧姞琛ヤ竵锛岃€屾槸 \(S_\eta+S_i\) 缁熶竴 PV 缃戠粶鐨勮竟鐣岃妭鐐癸紱鍐呴儴 \(S_i\) 鐨?PV anomaly 涓庤嚜鐢遍潰銆佽〃灞傚瘑搴﹀拰杈圭晫閫氶噺蹇呴』鍏卞悓婊¤冻鍚屼竴涓棴鍚堟潯浠躲€?
涓轰簡鎻忚堪娑℃棆鍊炬枩杩囩▼涓殑 PV 涓讳綋鍋忕Щ锛屽畾涔?PV centroid锛?
```math
\mathbf r_{PV,i}
=
\frac{
\int_{V_i}\mathbf r\,|PVA_i|\,dV
}{
\int_{V_i}|PVA_i|\,dV
}.
```

杩欓噷浣跨敤 \(|PVA_i|\) 浣滀负绗竴鐗堟潈閲嶏紝鏄负浜嗛伩鍏嶆璐?PVA 鍦ㄥ悓涓€灞傚唴鐩镐簰鎶垫秷銆傚悗缁彲杩涗竴姝ュ垎鍒畾涔夋 PVA centroid 涓庤礋 PVA centroid銆傜浉瀵硅〃灞傜殑浣嶆丁璺濆畾涔変负

```math
TD_{PV,i}^{\ast}
=
\frac{
|\mathbf r_{PV,i}-\mathbf r_{PV,0}|
}{
R_i
},
```

鐩搁偦灞傜殑浣嶆丁璺濆畾涔変负

```math
TD_{PV,i,i-1}^{\ast}
=
\frac{
|\mathbf r_{PV,i}-\mathbf r_{PV,i-1}|
}{
\bar R_{i,i-1}
}.
```

鍥犳锛屾湰鏂囧悗缁尯鍒嗕笁绫诲€炬枩涓績锛?
```text
velocity core tilt:
    閫熷害闆剁偣鎴栭€熷害缁撴瀯涓績鐨勫亸绉汇€?
completed-center tilt:
    鐢?completed centers 缁欏嚭鐨勮繛缁丁蹇冪粨鏋勫亸绉汇€?
PV-centroid tilt:
    鐢?PVA 涓讳綋璐ㄩ噺涓績缁欏嚭鐨勪綅娑¤窛鍋忕Щ銆?```

鑻?\(TD_{PV,i}^{\ast}\) 涓?completed-center tilt 鍚岀浉锛岃鏄庢丁鏃嬪€炬枩瀵瑰簲 PV 涓讳綋鏈韩鐨勫亸绉伙紱鑻ヤ簩鑰呬笉鍚岀浉锛屽垯璇存槑閫熷害鏍稿績銆佸嚑浣曟丁蹇冧笌 PV 寮傚父涓讳綋鍙戠敓鑴辫€︺€傚浜?complex 绫诲瀷锛岃嫢 \(TD_{PV,i}^{\ast}\) 闅忔繁搴﹁烦鍙樻垨 \(\epsilon_i^{PV}\) 寰堝ぇ锛屽垯鏇撮€傚悎瑙ｉ噴涓哄鏍搞€佹柇瑁傛垨杈圭晫閫氶噺涓诲锛岃€屼笉鏄崟涓€杩炵画娑℃煴鐨勫€炬枩銆?
## 5. Revised A锛氱粺涓€鏇查潰-灞傞棿鑰﹀悎 QGPV

鍦ㄧ \(i\) 涓唴閮ㄧ瓑瀵嗗害闈?\(S_i\) 涓婏紝revised A 鍐欎綔

```math
q_i^A
=
f_{n,i}
+
\frac{1}{f_0}\Delta_{S_i}\psi_i
+
\mathcal S_i^A.
```

鍏朵腑 \(\Delta_{S_i}\) 鏄彲鍙樺舰瀵嗗害娴佸舰涓婄殑 Laplace--Beltrami 绠楀瓙锛孿(\mathcal S_i^A\) 鏄粺涓€鐨勬洸闈?灞傞棿鑰﹀悎绠楀瓙銆備竴鑸舰寮忎负

```math
\mathcal S_i^A
=
C_{i-\frac12}
\left(\mathcal P_{i-1\to i}\psi_{i-1}-\psi_i\right)
+
C_{i+\frac12}
\left(\mathcal P_{i+1\to i}\psi_{i+1}-\psi_i\right).
```

杩欓噷 \(\mathcal P_{j\to i}\) 鎶婄浉閭绘祦褰?\(S_j\) 涓婄殑娴佸嚱鏁版姇褰卞埌 \(S_i\)銆傜涓€鐗堟暟鍊煎疄鐜伴噰鐢ㄥ悓涓€ \(x^\ast,y^\ast\) 缃戞牸涓婄殑鐩存帴鎶曞奖锛涘悗缁彲鍗囩骇涓虹湡姝ｆ洸闈㈡硶鍚戞姇褰便€?
瀵逛簬椤跺眰鍐呴儴闈?\(S_1\)锛屼笂閭诲眳涓嶆槸绌哄眰锛岃€屾槸琛ㄩ潰鑺傜偣 \(S_0=S_\eta\)锛?
```math
\mathcal P_{0\to1}\psi_0
=
\mathcal P_{\eta\to1}\psi_\eta.
```

鍥犳椤跺眰涓婅竟鐣岃€﹀悎鍐欎綔

```math
\mathcal S_{\eta,1}^A
=
C_{\eta,1}
\left(\mathcal P_{\eta\to1}\psi_\eta-\psi_1\right),
\qquad
C_{\eta,1}\sim\frac{f_n^{(1)2}}{f_0^2R_{\eta,1}^2}.
```

鍦ㄦ棤閲忕翰鍚堟垚鍧愭爣涓紝鏈€浣庨樁瀹炵幇涓?
```math
\mathcal S_{\eta,1}^{A\ast}
=
\left(\frac{R}{R_{\eta,1}}\right)^2
\left(\psi_\eta^\ast-\psi_1^\ast\right),
\qquad
\psi_\eta^\ast=\frac{g\eta}{UR}.
```

杩欒鏄庤嚜鐢遍潰鍜屽唴閮ㄩ《灞傛槸鍚屼竴涓?PV 鍙嶆紨缃戠粶涓殑鐩搁偦鑺傜偣銆傛棫鍐欐硶 \(- (R/R_{\eta,1})^2\psi_1^\ast\) 鍙槸鎶?\(\psi_\eta^\ast\) 榛樿涓洪浂鐨勯€€鍖栨儏褰紝涓嶈兘鐢ㄤ簬鏈?`adt_anom` 鐨勬暟鎹€?
瀵瑰簲鐨?baseline 浠嶄负鍥哄畾娣卞害骞抽潰褰㈠紡

```math
q_i^0
=
f_0+\beta y
+
\frac{1}{f_0}\nabla_h^2\psi_i
+
\mathcal S_i^0.
```

鍥犳 revised A 鐨勪慨姝ｅ彲鎷嗕负

```math
\delta q_i^A
=
\frac{1}{f_0}(\Delta_{S_i}-\nabla_h^2)\psi_i
+
\delta\mathcal S_i^A
+
(f_{n,i}-f_0).
```

绗竴鐗堟暟鍊奸獙璇佷繚鐣欎袱绫诲彲妫€鏌ヨ础鐚細鍐呴儴娴佸舰鏇查潰 Laplacian 淇锛屼互鍙婅〃闈㈣妭鐐?\(S_\eta\) 涓庨《灞?\(S_1\) 鐨勭粺涓€鑰﹀悎銆傚唴閮ㄧ浉閭绘祦褰㈤棿鐨勬樉寮忔姇褰卞樊鍏堜繚鐣欎负璇婃柇鍗犱綅锛屽悗缁啀鎵╁睍銆?
## 6. 灏忓潯搴﹀睍寮€

褰?\(S_i\) 鍙啓涓?\(z=Z_i(x,y,t)\)锛屼笖 \(|\nabla Z_i|\ll 1\) 鏃讹紝

```math
\Delta_g\psi
=
\nabla_h^2\psi
+
\delta\Delta_g\psi
+
O(|\nabla Z|^4).
```

浠?

```math
Z_x=\frac{\partial Z}{\partial x},\qquad
Z_y=\frac{\partial Z}{\partial y},
```

鍒欎竴闃舵湁鏁堢殑鏇查潰淇鍙啓浣?

```math
\delta\Delta_g\psi
\approx
-
\nabla_h\cdot
\left[
\nabla Z(\nabla Z\cdot\nabla_h\psi)
\right]
+
\frac12\nabla_h\cdot
\left[
|\nabla Z|^2\nabla_h\psi
\right]
-
\frac12|\nabla Z|^2\nabla_h^2\psi.
```

杩欎釜寮忓瓙鏄綋鍓?revised A 鏁板€煎疄鐜扮殑鏍稿績銆傚畠鎶婄瓑瀵嗛潰鏇茬巼銆佸潯搴﹀拰娴佸嚱鏁版搴﹁€﹀悎璧锋潵锛岀粰鍑虹浉瀵规丁搴?PV 鐨勫嚑浣曟敼鍙橀噺銆?

## 7. 鍊炬枩娑℃棆涓殑鐗╃悊瑙ｉ噴

璁炬瘡灞傛丁蹇冧负

```math
\mathbf r_{c,i}=(x_{c,i},y_{c,i}),
```

鐩稿琛ㄥ眰鍋忕Щ涓?

```math
\Delta\mathbf r_i
=
\mathbf r_{c,i}-\mathbf r_{c,0}.
```

鑻ョ瓑瀵嗗害闈笂鐨勫绉板瘑搴﹀紓甯镐负 \(\rho'_{s,i}\)锛屽€炬枩瀵艰嚧鐨勫钩绉诲睍寮€涓?

```math
\rho'_i(\mathbf x-\Delta\mathbf r_i)
\approx
\rho'_{s,i}(\mathbf x)
-
\Delta\mathbf r_i\cdot\nabla_h\rho'_{s,i}.
```

绛夊瘑闈綅绉绘弧瓒?

```math
\eta_{\rho,i}
=
-
\frac{\rho'_i}{\partial_z\bar\rho}.
```

鍥犳闈炲绉伴」鐨勪竴闃朵富瀵奸噺涓?

```math
\eta_{\rho,\mathrm{odd},i}
\approx
\frac{
\Delta\mathbf r_i\cdot\nabla_h\rho'_{s,i}
}{
\partial_z\bar\rho
}.
```

revised A 鐨勪綔鐢ㄦ槸杩涗竴姝ユ寚鍑猴細杩欑绛夊瘑闈㈤潪瀵圭О涓嶄粎鏀瑰彉 \(\eta_\rho\)锛岃繕閫氳繃 \(g_{ab}\)銆乗(g^{ab}\) 鍜?\(\Delta_g\psi\) 鏀瑰彉 \(q_i\)銆傛墍浠ュ€炬枩娑℃棆鐨勯€熷害鍝嶅簲涓嶅簲鍙粠鍥哄畾骞抽潰涓婄殑 \(\nabla_h^2\psi\) 鍒ゆ柇锛岃€屽簲姣旇緝

```math
q_i^0
\quad\text{and}\quad
q_i^A.
```

## 8. 鍩烘€佷笌鎵板姩鍋忕Щ

鍦ㄦ棤闄嗗湴銆佽繎浼煎叏鐞冨昂搴︾殑鐞嗘兂鑳屾櫙涓嬶紝绛夊瘑搴﹂潰鍙繎浼间负寰壈鐞冮潰銆傛洸闈?Laplacian 鐨勬湰寰佸嚱鏁版弧瓒?

```math
-\Delta_{g_i}Y_{\ell m}
=
\lambda_{\ell m}^{(i)}Y_{\ell m}.
```

鍦ㄧ悆闈㈡瀬闄愶紝

```math
\lambda_{\ell m}=\frac{\ell(\ell+1)}{a^2}.
```

鏈€浣庨潪闆舵ā鎬佷负 \(\ell=1\)銆傝嫢鍙栬酱瀵圭О鍩烘€侊紝

```math
Y_{10}(\phi)
=
\sqrt{\frac{3}{4\pi}}\sin\phi,
```

涓ゅ眰姝ｅ帇鍩烘€佸彲鍐欎负

```math
\psi_1^0=\psi_2^0=AY_{10}(\phi).
```

鏂滃帇鎵板姩瀹氫箟涓?

```math
\delta\psi_i=\psi_i-\psi_i^0.
```

鑻ユ壈鍔ㄨ灞€鍦版潈閲?\(W(\lambda)\) 闄愬埗锛屽苟涓旂 \(i\) 灞傜浉瀵归《灞傚瓨鍦ㄥ皬缁忓悜鎴栫含鍚戝亸绉?\(\Delta\lambda_i\)锛屽彲鍐欎綔

```math
\delta\psi_i
=
\sigma\epsilon
W(\lambda-\Delta\lambda_i)
Y_{10}(\phi).
```

灏忓亸绉诲睍寮€涓?

```math
W(\lambda-\Delta\lambda_i)
\approx
W(\lambda)-\Delta\lambda_i W'(\lambda).
```

褰撴繁灞傛枩鍘嬭皟鏁村彈鍩烘€佹祦骞虫祦涓斿瓨鍦ㄦ椂婊炴椂锛?

```math
\frac{\partial \Delta\lambda_i}{\partial t}
\sim
\frac{u_i^0}{R_{d,i}},
```

浠庤€?

```math
\Delta\lambda_i
\sim
\frac{u_i^0 t}{R_{d,i}}.
```

杩欑粰鍑哄眰闂寸郴缁熸€у亸绉荤殑鐞嗚鏉ユ簮銆俽evised A 灏嗚繖绉嶅亸绉昏繘涓€姝ユ槧灏勪负鏇查潰 PV 鍑犱綍椤癸紝鑰屼笉鏄妸瀹冭В閲婁负鍗曠函鐨勫浐瀹氬潗鏍囦几缂╅」銆?

## 9. 楠岃瘉鍙ｅ緞

姝ｅ紡楠岃瘉鍙瘮杈冧袱绉嶆ā鍨嬶細

```math
\text{baseline}: q_i^0,
\qquad
\text{revised A}: q_i^A.
```

楠岃瘉杈撳嚭搴斿寘鍚?
```text
qgpv_baseline_full
qgpv_model_A_full
qgpv_model_A_correction
pv_balance_residual
pv_centroid_x_R
pv_centroid_y_R
TD_PV_star
TD_PV_adjacent_star
```

涓嶅簲鍐嶇敓鎴愬叾浠栫粡楠屼慨姝ｅ垎鏀綔涓烘寮忔ā鍨嬪彉閲忋€?
閫熷害楠岃瘉鐩爣浠嶄负鐢熷懡鍛ㄦ湡鍚堟垚涓殑鍘绘皵鍊欐€佹丁鏃嬪紓甯搁€熷害锛?

```math
u' = u-\overline u_{\mathrm{doy}},
\qquad
v' = v-\overline v_{\mathrm{doy}}.
```

Morel 2019 瀵规湰鏂囩殑楠岃瘉鍙ｅ緞鎻愬嚭涓€涓澶栬姹傦細A model 鐨勬洸闈㈠嚑浣曢」蹇呴』鍚屾椂鎺ュ彈 PV 闂悎妫€鏌ャ€傝嫢鏌愪竴灞傛垨鏌愪竴鐩镐綅涓?
```math
\epsilon_i^{PV}
```

杈冨ぇ锛屽垯璇ュ眰鐨?\(q_i^A-q_i^0\) 鍙兘浣滀负 diagnostics 瑙ｉ噴锛屼笉搴旂洿鎺ヨ繘鍏ユ寮?QGPV forcing銆傛崲瑷€涔嬶紝鍑犱綍椤规槸鍚︹€滅湅璧锋潵鍍忊€濇丁鏃嬬粨鏋勫苟涓嶈冻澶燂紱瀹冨繀椤诲悓鏃舵弧瓒崇瓑瀵嗛潰鎺у埗浣?\(V_i\) 涓婄殑 PV 浣撶Н鍒嗕笌杈圭晫閫氶噺闂悎銆?
闄ら€熷害 skill 澶栵紝楠岃瘉杩樺簲姣旇緝涓夌鍊炬枩璇婃柇锛?
```text
completed-center tilt:
    TD_i^* from completed centers.

PV-centroid tilt:
    TD_PV_i^* from PVA centroid.

velocity response:
    u'/v' section and topview skill.
```

鑻?coherent 绫诲瀷涓?\(TD_{PV,i}^{\ast}\) 涓?completed-center \(TD_i^\ast\) 鍚岀浉锛屽苟涓?revised A 鐨?PV 闂悎娈嬪樊杈冨皬锛屽垯璇存槑鍊炬枩鏄?PV 涓讳綋鍜岀瓑瀵嗛潰鍑犱綍鍏卞悓鍋忕Щ鐨勭粨鏋溿€傝嫢 \(TD_{PV,i}^{\ast}\) 涓?completed centers 鑴辫€︼紝鎴栬€?\(\epsilon_i^{PV}\) 寰堝ぇ锛屽垯搴斾紭鍏堣В閲婁负澶氭牳銆佹柇瑁傘€佽竟鐣岄€氶噺涓诲鎴栨湭闂悎鐨勫嚑浣?proxy锛岃€屼笉鏄繛缁丁鏌卞€炬枩銆?
鑻?revised A 鐩告瘮 baseline 鍦?coherent 绫诲瀷涓彁鍗囷紝鑰屽湪 upright-like 绫诲瀷涓笉浜х敓鍚屾牱鎻愬崌锛屽垯璇存槑鏇查潰/绛夊瘑搴﹂潰鍑犱綍纭疄鏇撮€傚悎鎻忚堪鍊炬枩娑℃棆銆傝嫢 revised A 瀵归€熷害鍦轰笉鎻愬崌锛屼絾瀵瑰瘑搴﹂潰鎴?PV 鍝嶅簲鎻愬崌锛屽垯璇存槑璇ョ悊璁烘洿閫傚悎瑙ｉ噴绛夊瘑闈?PV 缁撴瀯锛岃€屼笉鏄洿鎺ラ棴鍚堝畬鏁撮€熷害棰勬姤銆?
## 10. QG 鏌辨丁鏈緛鍘熷瀷

涓轰簡鎶?Vi煤dez 鍨?Beltrami/Trkalian 绮剧‘娑′腑鐨勨€滄湰寰佹ā鎬佲€濇€濇兂杞寲涓烘洿閫傚悎涓昂搴︽丁鐨勮瑷€锛屾湰鏂囬噰鐢?QG 浣嶆丁鍙嶆紨绠楀瓙鑰屼笉鏄?curl 绠楀瓙銆傚父 \(f_0,N\) 涓嬶紝

```math
L\psi
=
\nabla_h^2\psi
+
\partial_z
\left(
\frac{f_0^2}{N^2}
\partial_z\psi
\right),
\qquad
\hat H\psi=-L\psi.
```

鍦ㄦ煴鍧愭爣涓紝鍒嗙墖鏌辨丁鍙啓涓?
```math
q'
=
\begin{cases}
-K^2\psi, & r<a,\\
0, & r>a.
\end{cases}
```

鍐呴儴婊¤冻 \(L\psi+K^2\psi=0\)锛屽閮ㄦ弧瓒?\(L\psi=0\)銆傚崟涓?\((m,k_z)\) 妯℃€佺殑鍐呴儴寰勫悜缁撴瀯涓?\(J_m(\kappa r)\)锛屽閮ㄨ“鍑忕粨鏋勪负 \(K_m(\gamma r)\)锛屽叾涓?
```math
\kappa^2=K^2-\frac{f_0^2}{N^2}k_z^2,
\qquad
\gamma=\frac{f_0}{N}|k_z|.
```

杈圭晫 \(r=a\) 涓婄殑 \(\psi\) 涓?\(\partial_r\psi\) 鍖归厤缁欏嚭绂绘暎/杩炵画娣峰悎璋辩殑鏈緛鏉′欢銆傜洿绔嬫丁鏌卞搴?\(m=0\)锛岀洿绾垮€炬枩鐨勪竴闃跺搷搴斿搴?\(m=1\) 浣嶇Щ妯℃€侊紝helical 鍊炬枩瀵瑰簲 \(\cos(\theta+k_z z)\) 鍨嬬浉浣嶆ā鎬併€傚畬鏁存帹瀵煎拰鍙绠楅獙璇侀噺瑙?`qg_cylindrical_eigenmodel.md`锛岄獙璇佸疄鐜拌 `src/Legacy/validation/qg_cylindrical_eigenmodel.py`銆?
## 鍙傝€冭祫鏂?
- Majda, A. J. and Wang, X. (2005). *Nonlinear Dynamics and Statistical Theories for Basic Geophysical Flows*.
- Vallis, G. K. *Atmospheric and Oceanic Fluid Dynamics*.
- Pedlosky, J. *Geophysical Fluid Dynamics*.
- Morel, Y., Gula, J., and Ponte, A. (2019). Potential vorticity diagnostics based on balances between volume integral and boundary conditions. *Ocean Modelling*, 138, 23-35.
- 鏈枃妗ｇ殑 revised A 鎺ㄥ渚濇嵁鐢ㄦ埛鎻愪緵鐨?`cuverlinear theory.docx` 涓€滄洸闈㈡ā鍨嬧€濅慨鏀规剰瑙佹暣鐞嗐€?