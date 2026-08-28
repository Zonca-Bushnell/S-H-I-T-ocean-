# QG 鏌辨丁鏈緛妯″瀷锛歅V 绠楀瓙鐨勭被閲忓瓙鍒嗚В

鏈枃缁欏嚭涓€涓敤浜庡€炬枩涓昂搴︽丁瑙ｉ噴鐨勭涓€鐗堣В鏋愭ā鍨嬨€傚畠涓嶄娇鐢?Beltrami/Trkalian 鏉′欢

```math
\nabla\times\mathbf u=k\mathbf u,
```

鑰屾槸鎶婂噯鍦拌浆浣嶆丁鍙嶆紨绠楀瓙浣滀负鏈緛绠楀瓙锛?
```math
q'=L\psi,\qquad
L=\nabla_h^2+\partial_z\left(S\partial_z\right),
\qquad
S=\frac{f_0^2}{N^2}.
```

鍦ㄥ父 \(f_0,N\) 鐨勭涓€鐗堜腑锛孿(S\) 涓哄父鏁般€傚畾涔夊舰寮?Hamiltonian

```math
\hat H\psi=-L\psi.
```

鑻?
```math
\hat H\psi=\Lambda\psi,
```

鍒?\(\Lambda\) 鏄?QG PV 绠楀瓙鐨勬湰寰佸€笺€傝繖涓被姣斿彧鐢ㄤ簬绠楀瓙銆佹湰寰佹ā鎬併€?杩炵画璋卞拰娉㈠寘锛屼笉鎶?\(\Lambda\) 瑙ｉ噴涓虹湡瀹為噺瀛愯兘閲忋€?
## 1. 鏌卞潗鏍囦腑鐨?QG PV 绠楀瓙

鍦ㄦ煴鍧愭爣 \((r,\theta,z)\) 涓紝

```math
L\psi=
\frac{1}{r}\partial_r(r\partial_r\psi)
+\frac{1}{r^2}\partial_{\theta\theta}\psi
+S\partial_{zz}\psi.
```

鍙栧崟妯℃€?
```math
\psi_m(r,\theta,z)=R(r)e^{im\theta}e^{ik_z z},
```

鍒?
```math
L\psi_m=
\left[
R''+\frac{1}{r}R'
-\frac{m^2}{r^2}R
-Sk_z^2R
\right]e^{im\theta}e^{ik_z z}.
```

鍥犳 \(-i\partial_\theta\) 缁欏嚭鏂逛綅妯℃€佹暟 \(m\)锛?\(-i\partial_z\) 缁欏嚭鍨傚悜杩炵画娉㈡暟 \(k_z\)銆傛病鏈変笂涓嬭竟鐣屾椂锛?\(k_z\) 鏄繛缁氨锛涘眬鍦版丁鏌辩敱澶氫釜 \(k_z\) 鍙犳垚 wave packet銆?
## 2. 鍒嗙墖鏌辨丁

浠ゆ丁鍗婂緞涓?\(a\)锛屽唴閮?PV 涓庢祦鍑芥暟鎴愭姣旓紝澶栭儴鏃?PV 寮傚父锛?
```math
q'=
\begin{cases}
-K^2\psi, & r<a,\\
0, & r>a.
\end{cases}
```

鍐呴儴婊¤冻

```math
L\psi+K^2\psi=0,
```

澶栭儴婊¤冻

```math
L\psi=0.
```

瀵瑰崟涓?\((m,k_z)\) 妯℃€侊紝鍐呴儴寰勫悜鏂圭▼涓?
```math
R_i''+\frac{1}{r}R_i'
-\frac{m^2}{r^2}R_i
+\kappa^2R_i=0,
\qquad
\kappa^2=K^2-Sk_z^2.
```

姝ｅ垯瑙ｄ负

```math
R_i(r)=J_m(\kappa r).
```

澶栭儴寰勫悜鏂圭▼涓?
```math
R_e''+\frac{1}{r}R_e'
-\frac{m^2}{r^2}R_e
-\gamma^2R_e=0,
\qquad
\gamma=\sqrt{S}|k_z|.
```

闅?\(r\to\infty\) 琛板噺鐨勮В涓?
```math
R_e(r)=C K_m(\gamma r),
```

鍏朵腑 \(K_m\) 鏄浜岀被 modified Bessel 鍑芥暟銆傜敱 \(\psi\) 杩炵画寰楀埌

```math
C=\frac{J_m(\kappa a)}{K_m(\gamma a)}.
```

鑻ヨ繘涓€姝ヨ姹傚緞鍚戦€熷害/鍘嬪姏姊害鎰忎箟涓嬬殑 \(\partial_r\psi\) 杩炵画锛屽垯寰楀埌
鏈緛鏉′欢

```math
\kappa\frac{J_m'(\kappa a)}{J_m(\kappa a)}
=
\gamma\frac{K_m'(\gamma a)}{K_m(\gamma a)}.
```

杩欏氨鏄垎鐗?QG 鏌辨丁鐨勨€滈噺瀛愬寲鈥濇潯浠讹細缁欏畾 \(a,S,K\) 鍚庯紝鍏佽鐨?\((m,k_z)\) 鐢辫竟鐣屽尮閰嶆畫宸喅瀹氥€?
## 3. 鍊炬枩妯℃€?
杞村绉板熀鎬佷负

```math
m=0.
```

瀹冭〃绀虹洿绔嬫煴娑°€傚皬骞呯洿绾垮€炬枩鍙互鍐欐垚鍩烘€佺殑姘村钩浣嶇Щ灞曞紑锛?
```math
\psi(r,\theta,z)
\approx
\psi_0(r,z)
-X(z)\partial_x\psi_0
-Y(z)\partial_y\psi_0.
```

鐢变簬 \(\partial_x\psi_0,\partial_y\psi_0\) 甯︽湁 \(\cos\theta,\sin\theta\)
缁撴瀯锛岀洿绾垮€炬枩鐨勪竴闃跺搷搴斿睘浜?\(m=1\) 鎵板姩銆?
铻烘棆鍊炬枩鍙互鐩存帴鍐欐垚 \(m=1\) 鐩镐綅妯℃€侊細

```math
\psi_1(r,\theta,z)=A(r)\cos(\theta+k_z z+\phi_0).
```

瀹冩弿杩伴殢楂樺害鏃嬭浆鐨勫亸蹇冩柟鍚戙€傜洿绾垮€炬枩鏇撮€傚悎娑″績杩戜技娌夸竴涓柟鍚戦殢娣卞害
鍋忕Щ鐨勪腑灏哄害娑★紱铻烘棆鍊炬枩鏇存帴杩?helical/鐩镐綅妯℃€併€?
## 4. Wave Packet

鏃犱笂涓嬭竟鐣屾椂锛屽瀭鍚戞尝鏁?\(k_z\) 杩炵画銆傚眬鍦颁腑灏哄害娑″彲鍐欐垚

```math
\psi(r,\theta,z)
=
\int \hat A(k_z)R_m(r;k_z)e^{im\theta}e^{ik_z z}\,dk_z.
```

鍏朵腑 \(\hat A(k_z)\) 鎺у埗鍨傚悜灞€鍦板寲銆佸€炬枩灏哄害鍜岀浉浣嶅叧绯汇€傝嫢
\(\hat A(k_z)\) 闆嗕腑鍦ㄦ煇涓?\(k_{z0}\)锛屽垯娑℃帴杩戝崟涓€ helical 妯℃€侊紱鑻?\(\hat A(k_z)\) 涓哄璋憋紝鍒欏彲褰㈡垚鍥寸粫鏌愪釜娣卞害 \(z_0\) 鐨勫眬鍦板€炬枩娑″寘銆?
## 5. 鍙绠楅獙璇侀噺

瀹炵幇浣嶄簬 `src/Legacy/validation/qg_cylindrical_eigenmodel.py`銆傜涓€鐗堥獙璇佷互涓嬮噺锛?
- 鍐呴儴娈嬪樊锛?  ```math
  L\psi+K^2\psi.
  ```
- 澶栭儴娈嬪樊锛?  ```math
  L\psi.
  ```
- 杈圭晫璺宠穬锛?  ```math
  [\psi]_{r=a},\qquad [\partial_r\psi]_{r=a}.
  ```
- 鍦拌浆閫熷害锛?  ```math
  u_g=-\partial_y\psi,\qquad v_g=\partial_x\psi.
  ```

杩欎簺閲忕敤浜庡垽鏂竴涓€欓€夋煴娑℃槸鍚︽槸 QG-consistent 鐨勮В鏋愬師鍨嬶紝鑰屼笉鏄敤浜?璇佹槑瀹屾暣 Euler 鎴?primitive-equation 瑙ｃ€?