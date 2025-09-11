import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import re
import string
import pandas as pd
from io import BytesIO
from datetime import datetime

# ------------------------
# Load model (local or online)
# ------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")  # <-- change if model is not local

model = load_model()

# ------------------------
# Categories (to be filled later)
# ------------------------
categories = {
    "Order Missing/ Pending Processing": [
        "order pending processing  1-115106826851",
        "order status pending processing in tm force since 25/8/25. iris reprot sd4753885 lodge on 28/8/25 until now still remain pend processing. pls assist asap. customer complained on late installation.",
        "1-cbeor26 - order ni masih belum ada dlm tmf",
        "2509000080460280 - mohon bantu ru inform tak nampak order dalam dalam tmf tapi order dah assign atas id dia  id : q104758 ru : mohd azizi bin mahzun annuar",
        "tak nampak order dalam dalam tmf",
        "tapi order dah assign atas id dia"
    ],
    "Order Next Activity Not Appear": [
        "tiada dlm rol..tq",
        "tiada dlm rol",
        "return tak masuk dalam bakul lobs/cxm",
        "done mir?",
        "order return, next activity tak appear",
        "salam team,  trigger semula order ini di page rrol. activity mir di siebel masih in progress 1-114085556792",
        "not in rol | hi team  order returned due to customer tapi tak ada next owner acitivty. tqvm",
        "salam, order tiada dalam rol utk ra. 1-115950242200. tq",
        "next activity x appear",
        "....2508000079730082 order ni tiada di scheduled page tp ada di page order search",
        "no pending user, tq  2509000080365841",
        "00378065688 - order 1-cbi3c7w 00378035878 - order : 1-cbhvjqo  mhn bantuan masukkan semula dalam oal...ui tidak nampak order sebab mdf sudah done activity remove mdf jumpering...tq",
        "masukkan semula dalam oal",
        "npua",
        "order npua",
        "tiada dalam oal",
        "tiada dalam rol ..mohon bantu.",
        "slot 10/9 tapi mir masih ip",
        "salam 2509000080381065 masukan order dalam basket jcc. tq",
        "order tiada di schedule page 1-115878860659 tq"
    ],
    "CC Not Appear": [
        "cc not appear",
        "bantu cc x kuar",
        "cc tak appear 1-114576765037"
    ],
    "Order Returned but Unscheduled": [
        "2509000080687620 mohon bantuan order return jadi unscheduled",
        "bantuan order return jd unscheduled",
        "order ra date hari ini jam4 pm..order jadi unscheduled 2509000081041721"
    ],
    "Order Capping Issue": [
        "bukakan capping utk id q003975 nak slot whp 14 line 1 lokasi (cust minta buat semua harini)",
        "order gomen kementerian sumber manusia  dlm portal tmf ui q003975 ada dpt 13 order. tp ui hanya ada 10 order shj 1-cbjdpgv 1-cbjzqxz 1-cbjx85i 1-cbjy128 1-cbjz0vp 1-cbjz165 1-cbjx8k4 1-cbjy0o0 1-cbjx22i 1-cbizjkr"
    ],
    "RG6 - RG7 Equipment Info Update": [
        "equipment rg7 ke rg6 ax3000 (router dan juga mesh)",
        "tukar equipment ke rg6 2.5",
        "(1gbps) ... mintak tukarkan tagging wifi 7 kepada wifi 6",
        "tukarkan ax3000 kepada rg7",
        "mintak tukarkan equipmnt rg7 ke combo 2.5g skli mesh. tq",
        "combo rg7 ke rg6 order 1gbps unc30val2412065163 unc30val2412065169",
        "salam kak mintak tukar dari wifi7 ke wifi6 unc30val2412061878",
        "unifi home 2gbps with netflix bantuan tukar rg7 ke rg6",
        "mohon bantu tukarkan rg vdsl single box kepada rg 7..tq",
        "1.)order schoolnet- mohon bantu tukar rg7 ke existing rg6 - combo ax3000",
        "order schoolnet- mohon bantu tukar rg7 ke rg6 - combo ax3000",
        "equipment rg dan mesh... dari rg7 ke rg6 combo 2.5g",
        "brg dari rg7 ke combo ax3000 4.zone 5 : bw",
        "flag masih rg7 1-115972738401",
        "flag masih rg7, ru pemasangan pakai rg6  1-115559812996",
        "1-115109708172    mohon bantu tukar cpe kepada rg6",
        "2509000080219214 | isu fail update equipment  mohon bantuan untuk update cpe wifi 7 ke wifi6 ax3000 combo.  no order: 2509000080219214 s/n no 1: unc25val2412025438 s/n no 2 mesh: unc25val2412025439",
        "tma q104706 1-113191017898 mntk tlg tukar eq ke combo biasa, order schoolnet. tq",
        "mohon bantuan tukarkan equipment rg7 ke rg6 ax3000 router & mesh....order modify 1gb  2509000080869301",
        "mohon bantuan flag masih rg7 ru x lepas nak donekan order..cust sudah nak keluar rumah  1-115865565435",
        "1-115874070879 mohon bantu. masih error utk done. mohon bantu trigger wifi6",
        "trigger wifi6",
        "updated equipment pda rg6 ax3000..order 1gb",
        "ubah ke wifi 7 .customer behaviour. order downgrade ke 1gb. customer tetap nak wifi 7 rujukan dr cff"
    ],
    "New/ Existing/ Delete Equipment Info Update": [
        "semakan order upgrade pakej 2gb tp dlm tmf equipment yg new vm shj",
        "tukarkan vm kepada rg6 tq",
        "cpe new ke existing, order force done cancel - iris error",
        "tukar wifi(rg) jdi new equipment atau existing.order hanya tambah mesh tpi rg dh delete .ui tak boleh install sebab status rg delete sd4773474",
        "tukar servis poin ke combo box",
        "order tmforce team id: order date: 8/9/2025 order no: remark: cust apply upgrade 1gbps, tapi dalam tmf hanya tukar modem. router sedia ada tak support 1gbps. mohon tukar equipment ke combo wifi 6"
        "relocate bantuan ui x blh nk done order, equipment dlm tmf adalah existing  tp cpe berlainan.. mesh rg6 dlink, uonu ax3000 skyworth",
        "tukarkan  ata jd existing sbb kat site customer dah ada existing ata 4 port"
        "sp ke combo rg6 ax3000",
        "bleh check dk oder ni.dkleh nk complete..nk scan combo xleh",
        "ru xleh scan cpe combo..dh try log out login pun masih sama",
        "bg isu order ini  1-115954384256 order ni cust cadari kap baru tukar rg6 combo minggu lepas ..harinhi keluar oder modify tukar onu pulak..blh check blik tk order ni  tq"
        "order ni boleh tukar equipment dari rg4 ke rg6 ax3000 x... upgrade 500mbps",
        "existing equipment combo box be7200 wifi 7 ke combo box ax3000 2.5ghz 4.zone : pgc",
        "tukar mesh ke existing ye, order schoolnet ( sekolah dah ada existing mesh )",
        "tukarkan equipment serv point vm ke new combo rg7 order modify 1gb  1-115960101049",
        "tukar equipment new mesh wifi ke combo rg6, modify 500mbps  2508000079549409",
        "tukarkan existing service point ke new . ru maklum model ni tak  support universal..order relocate",
        "1-115865470728 modify 09.30  mhn bantuan tukarkan new sp ke new combo (rg6) order modify 1g...tq",
        "1-115791690115 modify upgrade 1gb - mohon bantuan tukar rg7 ke rg6 ax3000 - tukar existing rg4 ke new rg6 ax3000 - delete new service point",
        "1-115803658502 mohon bantu.. service point masih existing",
        "mohon bantu.service point masih new.mohon tukar existing & add new rg combo box 1-115965608879",
        "mohon bantuan add new combo box. order modify infra dari vdsl ke ftth tq  1-115625268896",
        "mohon bantu jadikan cpe existing. order relocate guna cpe asal: 1-115392918697",
        "jadikan cpe existing",
        "order relocate guna cpe asal",
        "1-115874174567 modify upgrade 1gb - delete new service point",
        "delete new",
        "delete new service point",
        "modify upgrade",
        "tukarkan  8 dect (new cpe) kpd existing krn cstmer guna pabx tq",
        "minta tukar btu ke new..order x share btu",
        "boleh check ini equipment sdh add 1 ata pun masihh sama",
        "oder. ni 1gb tp tukar modem sj btolke",
        "jdkan new btu dekat equipment..order relocate sebelum ini cust pakai copper (sbvm)",
        "add new sp, dekat dlm order tak de sp",
        "add modem...order modify infra.",
        "updated order 2gb guna combo wifi 7.dlam tmf ada modem shj,.",
        "tukar btu ke new..order x share btu -1-116438099969",
        "tukar sp kepada existing, order relocate btu masih baru",
        "mintak existing cpe onu order relocate",
        "order modify tolong tukar eqiupment new ke existing",
        "eqiupment new ke existing"
    ],
    "Update Order Equipment Details": [
        "nk done keluar error seperti berikut..tq router-rg6dlax32311015973 mesh rg6-rg6dlax32311015977",
        "order nak done tap error sn baru 👉 unc30val2501038792 ....tq",
        "order modify add no shj x dapat nak done order,ade error pink.  order:2509000080361602  del:cpogelgc2501002409   onu:cw7220al2508010552 cw7220al2508010553  ata:atalc4nr2501008265",
        "ru tak boleh done, order add new ata sahaja, tak tukar cpe lain, tak boleh done, sn ata new: atanov042505000248",
        "intelligence netcare sdn bhd mohon bantu ui tak dapat complete order,order delete fix ip,cust pakai cpe sendiri tak guna cpe tm.tq",
        "bypaskn rg dgn mesh,main tak sama nak done tak lepas"
    ],
    "Unable to Swap Number": [
        "1-c1z5awa order whp tidak dapat swap number ...sudah call ftc dia suruh refer jcom"
    ],
    "Bypass Extraport": [
        "mhn bantuan bypass xp",
        "bantuan .. bypass fdp clensing ..dekat dp ada reading .. dalam tracebite tkde reading ..",
        "mohon bypass extraport. iris access denied",
        "1-115636242567 tolong bypass extraport lhd_c001_dp0071 09out",
        "iris error.. xboleh log iris  2509000081023987 mohon bantu bypass ep.no order tiada dlm apps/troika.ep flag y"
    ],
    "Bypass HSI": [
        "minta bypass speedtest dan wifi analyzer sbb oder force done",
        "1-112885939768 hsi failed - order fdc .. ble bantu bypass iris nak balik awal kali"
    ],
    "Manual Assign Button not Appear": [
        "tiada button manual assign",
        "hsba bantuan manual slot x appear",
        "manual assign x appear",
        "salam, mohon bantu order tiada button manual assign",
        "manual assign tak appear",
        "1-c1z5awa button manual assign tidak appear tqvm",
        "team mohon bantu tiada button ma. cxm info sdh done cc 1-114890055872",
        "1-114089879333 mohon bantuan order ra tidak boleh manual assign",
        "dear all mohon bantu 1-112893674327///jabatan sukarelawan malaysia rela negeri sarawak order pada 04092025. tiada aktivi manual assign. failed to assign. sd4769621",
        "1-cbc64pd ma not appear",
        "ma not appear",
        "team mohon bantu..tiada button ma. cc sdh done",
        "1-cbjz7uk ma not appear",
        "ma"
    ],
    "Order Error 500": [
        "Tak boleh nk return",
        "bantuan id Q102829 x blh nk return order,ui dh try relogin pun sama"
    ],
    "Invalid ICBRN Number": [
        "no ssm/ic cust, dlm tmf tiada",
        "minta ic no untuk order ni.. xd dekat detail customer",
        "bantuan ui fail utk update ic/br no",
        "ic dan ssm no. dah masuk tp invalid, tak lepas done.",
        "order x bole done ic cust invalid.",
        "team sdh masukkan no ic sperti penama didalam order tetapi tidak dapat"
    ],
    "Invalid Order Segment": [
        "segment  kepada sme. order s10. tq 2509000080543072",
        "2509000080425247 mohon bantuan segment kepada sme. segment yang betul adalah sme",
        "order ni kepada segment sme, order biz"
    ],
    "Update Contact Number": [
        "update ctc no",
        "team, mohon bantu 1-116379701877 update ctc no 0166855599 di tmf. thanks"
        "team, kindly assist to update ctc 0139433051  for order id :2508000078626166 . tq.",
        "team, mohon bantu utk tukar hp contact number 0129898252 utk order 2508000079145394, tq",
        "2509000080420382 – pls change contact no. to 012-8461870. tq.",
        "tukarkan contact number kepada 01139851691",
        "update contact number.....tq order:  2508000078206670 / sw food & bar concepts sdn bhd ctc no: 0166812312",
        "update contact number. ....tq order:  2508000079407460 / yap mun kit contact no mobile: 01112240477 contact no home : 0177060128"
    ],
    "Release Assign to Me": [
        "dapat error nak ra order",
        "x boleh book appt, keluar error seperti di atas...tq",
        "release order ""asssign to me""...team maklum nk release tp hilang dlm tmf...tq"
    ],
    "Order D&A In-Progress": [
        "task d&a ip",
        "d&a masih in progress",
        "d&a ip",
        "dna ip",
        "d&a masih ip",
        "assalamualaikum 1-116382418268 - d&a masih in progress, mohon bantu"
    ],
    "Unsync Order": [
        "ui maklum dah done tapi nampak in progress lagi dlm tmf. tk",
        "2508000079434474 nak tanya knp takde button book appt. (nak retime...)",
        "ui kata xde button utk on site. tk",
        "mntk tlg tngk order dah done tpi masih in progress"
    ],
    "Revert BAU SWIFT-TMF Order": [
        "1-cbga8gd | assalam team, mohon revert order id dibawah ke tmf system. urgent!!! tq 1-cbga8gd 1-cbgnu4p",
        "1-cbeoqy6 - status completed done in tmf 3/9/2025 tq 1-cbeub0k - status completed - swift tq 1-cbeor26 - masih tiada di subscriber activity list swift @ tmf"
    ],
    "Reopen Proposed Cancel Order": [
        "order status propose cancel - utk ra semula - tq",
        "status order propose reappt......order ada di page order search tapi tiada di rrol",
        "2508000079953130 | hi team , tarik balik order ini from propose cancel pool, order can be installed update by ru team. tqvm team",
        "order proposed cancel 2508000079156490 bantuan... order proceed pemasangan..."
    ],
    "Order In-Progress but Auto Done": [
        "tiba done sendiri masa tengah nak masuk attachment . detail speedtest semua takmasuk pun lgi sbb nk bypass btu",
        "terus ke saf..x sempat scan cpe lg, tq",
        "ui dah complete order tapi masih x hilang. ui xleh nak buat next order. tk"
    ],
    "TT RG5 Equipment Update": [
        "tukar cpe rg5 & onu ke new router combo",
        "ada error untuk done  1-116311485233  sn router : vdlf24dl2111011465",
        "salam team x dapt close ctt untuk pertukaran router rg5 minta bantuan  ctt no :1-116329668255 package : 100mbps sn lama :vdlinnne2007051543 sn baru : vdlf24dl2111009558",
        "tak boleh tukar rg 5..equitment combo.. tapi cust pakai rg 5"
    ],
    "TT RG6/ Combo Update": [
        "salam team x dapt close ctt untuk pertukaran router minta bantuan  ctt no : 1-116315464203 package : 100mbps sn lama : rg6fhax32208032145 sn baru :unc30val2410080579",
        "sn rg lama:rgxaztwr2205016652 sn combo baru : unc30val2410081417 no ctt:1-116314999694 package :100mbps mohon bantu tm point telah janjikan set combo tq ya",
        "assalammualaikum, mohon bantuan untuk error ni.   id :q103945 tt number:1-26818262517 old sn:whps20al2405000239 new sn:whp910dl2501001033",
        "salam,mohon tukar tagging,cust pekej 300mb  info: ctt no: 1-116182596740 old rg sn: rgxtplc12102000715 new rg sn: uncfh5f32411019303",
        "info:tukar rg5 ke combo 100 mbps ctt no:1-115950652482 old rg sn:rgxaztwr2207005074 new rg sn:uncfh5f32411032691",
        "remark : mohon bantuan tukarkan flag ke combo box dan mohon update acs bind.*  ctt: 1-116270632518 s/n cpe lama: rgxtplec2205052950 s/n cpe baru: unc30val2411098979 login id : tey6686@unifi  terima kasih 🙂",
        "assalamualaikum & salam sejahtera,  remark : mohon bantuan tukarkan flag ke combo box dan mohon update acs bind.  ctt: 1-116311140378 s/n lama: rgxaztwr2112008593 s/n baru: unc30val2411098930 login id : norsitikamsiah@unifi  terima kasih 🙂 @~taufik z.",
        "remark : mohon bantuan tukarkan flag ke combo box dan mohon update acs bind.*  ctt: 1-115722470135 s/n cpe lama: rgxperdl1709003741 s/n cpe baru: unc30val2411098813 login id : isminorhanbinismai@unifi  terima kasih 🙂 @~taufik z.",
        "remark : mohon tukarkan flag ke combo box ctt:  1-115865956193  s/n lama: rgxtplc11812010132 s/n baru: uncfh5f32502024708 login id: zaynvivian95@unifi terima kasih",
        "remark : mohon bantuan tukarkan flag ke combobox dan mohon update acs bind.  ctt: 1-116312296513 s/n lama: rgx825dl2012009971 s/n baru : unc30val2501010999 s/n baru (gpon) : cwtcb81ad76c login id : suhairina08@unifi",
        "remark : mohon tukarkan flag ke combo box ctt:  1-115976383308  s/n lama: rg6fhax32208019428 s/n baru: unc30val2411014648 login id: honghai@unifi terima kasih",
        "1-115970926445 minta update flag tukar ke combo box. customer sudah tukar pakej ke 300m.",
        "mohon tukar tagging ke combo/rg6 no ctt:  1-115969916464 sn lama:  rgxdvgdl2109025080 sn baru:  unc30val2412069248",
        "ctt :1-116313934089 sn lama:rgxtplc11903028849 sn baru:rgx835dl2307001679 pakej:100mbps",
        "enable combo flag 1-116316769444 old sn : rgxaztwr2105049908 new sn : unc30val2412061412",
        "1-115972600583 old sn : rgxperdl1705008311 new sn : unc30val2501002986",
        "truckroll cpe replacement  1-116315005124 old rg rgxdvgdl2109013485 newrg unc30val2501056851",
        "nnt berikan detail spt dibwh: ctt no: nnt berikan detail spt dibwh:  ctt no: 1-116000429645 old rg sn: rgx830dl2109018545 new rg sn: unc30val2501034470 remark : serial number does not exist in your cpe ( truckroll )",
        "ctt no :  1-116323270125 old rg :  vdlaztds2008003140 new rg : unc30val2412066279  mohon enable flag",
        "mintk tukar flag tt no:1-116321092138 new sn:uncfh5f32411001561 old sn:rgxaztwr2207027329 tt truckroll",
        "info: ctt no:  1-116312914201   old rg : rg6dl60z2206082971  new rg sn: uncfh5f32411022966    https://forms.gle/irzc9fmbnz8yzgtva  team support, mohon bantuan untuk enable flag hanya selepas bukti form submission disertakan.  terima kasih atas kerjasama semua.",
        "mohon bantu kembali, ui update cpe bertukar ke rg5, req update ke rg6  order no:1-112022311713 minta bantuan utk tukar wifi 5 kepada wifi 6",
        "1-115880410969..xlepas nk replace combo",
        "ctt - 1-11588495833 sn lama - rgx835dl2307000024 sn baru - unc30val2501040315  bantuan - customer setuju renew contract... dh dijanji kan oleh tm point..  tq",
        "no ctt:1-115787315701 old ce:com60vdl2304001102 new cpe:cw7220al2508011852",
        "1-116380461845 | salam, mohon tukar tagging,cust pakej 100mbps. sudah tiada stock rg5 dan sbvm.  info: ctt no: 1-116380461845 old rg sn: rgx842dl2107015642 new rg sn: uncfh5f32412032170",
        "sudah tiada stock rg5 dan sbvm",
        "mohon bantuan..1-115961396302..tukar onu,xboleh nk done tt..sn onu lama onunokia1803010458..sn baru uonwfh2a2505004787",
        "mohon bantuan @en md rad mohon bantuan ctt no :1-116321367562 no siri lama :vdlinnne1911037019 no siri baru  :unc30val250134169",
        "salam bantuan team tak boleh tutup ctt tk boleh swap cpe  team tukar combo box  ctt no 1-116375717449",
        "ctt tk boleh swap cpe  team tukar combo box",
        "ada error untuk done  1-116336590617  sn baru : uncfh5f32502020536 sn lama : rg6wfh1y220701631",
        "ada error untuk done",
        "1-116325374545 - tt ni sblm ni pd 13/8/25 tukar rg6 & mesh wifi kepada combo box & mesh wifi.. customer pilih option renew contract..dlm rekod tt yg keluar balik masih cpe lama rg6..mohon bantuan untuk update combo box baru dlm equipment customer",
        " salam team x dapt close ctt untuk pertukaran router combo minta bantuan  ctt no :1-115980922928 package : 300mbps sn lama :vdlinnne1911015109 sn baru : unc30val2412063318",
        "info:100 mbp combo ctt no: 1-116384133328 old rg sn:vdlaztds2109009572 new rg sn:uncfh5f32411042436 ctt save program dr telemarketing request renew contract tukar router wifi 6",
        "ctt no : 1-116444069593 rg old : rgxtplc11901017459 rg new : uncfh5f32411020443",
        "no tt:1-116480219495 remark: ctt truck roll cust tukar ke combo bantuan tukar flag",
        "remark: ctt truck roll cust tukar ke combo bantuan tukar flag",
        "assalamualaikum & salam sejahtera, tuan/puan zone ctt coordinator/sp/exec untuk setiap permohonan untuk enable flag rg6 di dalam tmf, mohon berikan maklumat seperti di bawah & mengisi google form yang disertakan (link terkini) & lampirkan screenshot form submission di sini.  info: ctt no: 1-116443193056  old rg sn: rgx835dl2307011663  new rg sn: unc30val2501028768  https://forms.gle/irzc9fmbnz8yzgtva  team support, mohon bantuan untuk enable flag hanya selepas bukti form submission disertakan.  terima kasih atas kerjasama semua.",
        "minta update  no ctt : 1-116483225947 sn lama : rg6gleex2204026011 sn baru : unc30val2501033247  mesh lama : rg6gleex2204025382 mesh baru : unc30val2501033248"
    ],
    "TT V1P": [
        "assalamualaikum..tt v1p 1-26814783348 appt semula hari ni 10.30 am..tq",
        "appt v1p @ 4.30pm 1-26818203485",
        "00506916920 1-26817775658 team mohon bantu utk appt ctt ni hari ini pukul 14:30 tq",
        "team minta buat appt pukul 16:30 hari ini 00506724003 1-26818262517",
        "slot 1-26817279018 @ 4.30pm",
        "1-26815105984 appt 3.30pm  tq",
        "tt v1p 1-26814783348 appt semula hari ni 10.30 am..tq",
        "appt v1p @ 4.30pm 1-26818203485",
        "00506916920 1-26817775658 team mohon bantu utk appt ctt ni hari ini pukul 14:30 tq",
        "1-26819223052 bantuan ra 2.30pm, tq",
        "ra tt v1p 1-26819345735  3:30 pm",
        "dear team apptkan v1p 1-26818264514|appt 330pm 🌾terima kasih🌾",
        "appt v1p 1-26818331322 430pm",
        "1-26819047788 bantu ra 11:45 am",
        "1-2",
        "bantu ra",
        "bantu ra am pm",
        "1-26820223665 bantuan v1p slot 11am",
        "mohon bantuan ra appt ctt v1p @230pm. tq 1-26819047743",
        "1-26813027589 ra 2:30pm",
        "ra ctt v1p 4:30 1-26821452755",
        "ctt 1-26821375701 / v1p mohon bantu slot appt tt v1p 4.30pm...tqvm",
        "slot",
        "bantu slot",
        "bantu slot appt",
        "bantu slot appt tt v1p"
    ],
    "TT Error 400": [
        "slot appt",
        "1-115955192035 ada eror 400. tq",
        "1-115961233664, xleh slot appt",
        "error 400 tq",
        "ctt 1-116394965125. error 400 tq",
        "no slot 1-116330643385, tq",
        "appointment error (400). tq  1-116374056595",
        "salam, ctt 1-116345228405 x keluar slot book appt",
        "tdk bole slot appt, error 400",
        "semakan 1-116487978265, ada eror 400. tq",
        "error 400",
        "err 400",
        "eror 400",
        "1-116493162085, tjb_v1011_0001, ada error 400 , tq",
        "ada error 400",
        "ada eror 400",
        "failed to manual slot id"
    ],
    "TT HSBA Reappointment": [
        "untuk patchkan ctt tq 1-116312148935 - appt on 10/9/25 at 11.30am",
        "patch ctt 1-116317835775 pada 11/09/2025  @ 11:30 am",
        "patch ctt dibawah :  1-116320728665 - 09/09/2025 @ 02:30:00 pm",
        "1-115504698155 - team patch appt on 09/09/2025 10:30:00 am. tq",
        "salam team, 1-116326647785 -patch slot 12/09/2025 09:30:00 am.tq",
        "utk patch ctt dibawah :  1-115973031505 - 09/09/2025 @ 04:30:00 pm  1-115804178865 - 09/09/2025 @ 02:30:00 pm  1-115789688125 - 09/09/2025 @ 09:30:00 am",
        "pacth ctt 1-116385142175 12/09/2025 11:30:00 am tq"
    ],
    "TT Missing": [
        "tt 1-115443089895 tiada dalam tmf",
        "ctt missing slps return fs troubleshooting tq 1-116065619936",
        "ctt missing. mohon bantuan. 1-116311028092",
        "tt 1-115443089895 tiada dalam tmf",
        "bantuan ctt missing 1-116139913645",
        "ctt missing",
        "team.....mohon trigger tt 1-116324250905....coz telah missing dlm activity list tmf......tq......zita.",
        "tt missing. mohon bantuan. 1-116132927093",
        "trigger ctt ini missing",
        "missing di tmf"
    ],
    "TT Unsync": [
        "ctt status cancel dlm nova masih appear dlm tmf  1-116258282215",
        "tt cancel tapi masih appear",
        "team nak close ctt x boleh",
        "tak boleh colse ctt",
        "tt 1-112719548255 not sync tmf resolve, nova ip",
        "bntu clearkn tmf nova cancelled",
        "ctt sudah closed di nova"
    ],
    "Update Granite Network Details": [
        "1-116311380548 - mohon betulkan dp id tin-1-d6qy8p3-001 kepada tin_c006_db0034",
        "1-115958165689 tiada detail cab",
        "minta bantuan adv ctt 1-115970783375 tak boleh tukar fdp dalam granite di tmf",
        "mohon bantu ctt tiada detail kabinet 1-116322359307",
        "1-116322359307 | mohon bantu ctt tiada detail kabinet 1-116322359307",
        "tt 1-116383768545 granite tiada data",
        "update cabinet/fdp dalam tmf,fizikal plgn berada di kgu_c002_db0012..tq  1-115861232625 kgu_c002_dp0009a - tmforce kgu_c002_db0012 -granite"
    ],
    "Update DP-Cabinet Location": [
        "salam team,  mohon bantuan update dp location order digi ni 1-115819132630 daripada on pole kepada building floor.",
        "daripada on pole kepada building floor",
        "dekat granite dp type street cabinet tapi di network on pole. dekat site dp type yang betul adalah street cabinet"
    ],
    "TT Error 400": [
        "1-115733638005 | salam team,  mohon bantuan tidak dapat view slot ctt: 1-115733638005 dp id:  mti_c102_dp0001",
        "salam tmf team, @normah mohd salleh @kak nurul mohon bantuan untuk issue ctt ni team: 1-114491447905 missing cab di granite di tm force. ctt : 1-114491447905 cab : swy_c019 dp : swy_c019_dp0017",
        "missing cab di granite di tm force",
        "1-116379866205 ult_c010_dp0050 mohon bantuan no slot",
        "bantuan tt slot",
        "bantu ctt no slot"
    ],
    "TT Duplicate Activity": [
        "cancel 1 activity id 1-116247023935  a-0009903201 a-0009903196",
        "cancel act duplicate 1-116315708835 a-0009908381"
    ],
    "TT TMF-Physical CPE Unsync": [
        "sn rg fizikal dengan tmf tak sama  ctt : 1-116245831767 fizikal : 159fe54e0057211b0 tmf : 0405202300000",
        "1-115885719055 sn onu fizikal - uonwfh2a2312002473 sn onu dalam list - uonwfh2a2312000612 bantu update list equipment.",
        "sn rg fizikal tak sama dengan tmf no ctt : 1-116153828615 rg sn fizikal : rgx842dl1903020843 sn tmf : rgwtull71309006364",
        "tolong update service point punya sn. sebab dalam tmf dengan fizikal tak sama. tq  no ctt : 1-116248891495 sn : onualual1706002509 mac : 9c50eee65894",
        "add onu dalam equipment di tmf, team inform customer ada onu di fizikal  1-116313069961  sn : uonzteh92505209321",
        "1-116313069961 mohon bantuan untuk add kan onu dalam tmf.  sn no : uonzteh92505209321",
        "1-116316069495. minta bantuan git add onu lama dlm order sbb perlu tukar onu sahaja.  onu lama : onualual1603007277 onu baru : uonzteh92505201369",
        "ctt no: 1-116045080136 fizikal onu : 4857443d6927a9a mohon bantu update sn onu lama cust",
        "1-116330386115 tt ni cust takde mesh bila nak update keluar mcm ni mohon bantuan",
        "tmf , tt ni nak tukar stb, tapi tak ada dalam equipment info. kat nova masih active hypptv. mohon bantu",
        "dalam equipment tmf tiada onu tapi di premis existed.  1-116376046915 uonwfh2a2506008576",
        "customer nie fizikal ada 2 sahaje combo. unc25val2412013327 yang nie tak wujud fizikal. minta remove dri list equitment customer",
        "xboleh swap combo, combo asal skyworth, saya tukar dgn skyworth same brand  1-116321727895",
        "xboleh swap combo, combo asal skyworth",
        "verify phone dalam tmf tiada tapi customer inform pernah dapat phone dari tm  1-116327647495",
        "cust no tel isu ata xde kat site dlm tmf ade cmne nk wat",
        "team xboleh update cpe..",
        "x dapt close ctt untuk pertukaran  cpe..dlm tm force cpe list btu tiada..team nak scan btu tak boleh",
        "verify, customer ada langgan mesh sekali ke ? sebab team inform di fizikal ada mesh tetapi system tiada"
    ],
    "TT - LR Linkage": [
        "salam team bantuan clear tmf ctt link lr20250312-39719 1-106191315217 1-106194182478 1-106312056435 1-106312232721 1-106346917798 1-106471083514 1-106678032670 1-109407463175  tq",
        "ctt unlink 1-115388049295 tiada dalam tmf.."
    ],
    "TT - Activity Work Type Blank": [
        "mohon bantuan...ctt tiada activity work type...ctt no : 1-115880486755..iris sudah closed minta retry...tq"
    ]
}

# Build initial embeddings (case-insensitive)
category_embeddings = {
    cat: model.encode([s.lower() for s in sentences], convert_to_tensor=True).mean(dim=0)
    for cat, sentences in categories.items()
}

# Threshold
SIMILARITY_THRESHOLD = 0.52
new_groups = {}
group_counter = 1

# ------------------------
# Regex for ticket/order/ID
# ------------------------
ID_PATTERN = re.compile(r"(?<!\w)(?:1-[A-Za-z0-9]+|250\d+|Q\d+|TM\d+)(?!\w)", re.IGNORECASE)

def has_valid_id(msg: str) -> bool:
    return bool(ID_PATTERN.search(msg))

def extract_ids(msg: str):
    return ID_PATTERN.findall(msg)

# ------------------------
# Categorization helpers
# ------------------------
def clean_message(msg: str) -> str:
    # Remove ticket/order IDs
    msg = ID_PATTERN.sub("", msg)
    # Lowercase
    msg = msg.lower()
    # Remove punctuation
    msg = re.sub(rf"[{re.escape(string.punctuation)}]", " ", msg)
    # Collapse multiple spaces
    msg = re.sub(r"\s+", " ", msg)
    return msg.strip()

def categorize_message(msg):
    global group_counter

    clean_msg = clean_message(msg)  # normalize before encoding
    emb = model.encode(clean_msg, convert_to_tensor=True)

    scores = {
        cat: util.cos_sim(emb, emb_cat).item()
        for cat, emb_cat in category_embeddings.items()
    }
    if scores:
        best_cat, best_score = max(scores.items(), key=lambda x: x[1])
    else:
        best_cat, best_score = "Uncategorized", 0

    if best_score >= SIMILARITY_THRESHOLD:
        return best_cat, best_score
    else:
        new_cat = f"auto_group_{group_counter}"
        new_groups[new_cat] = emb
        category_embeddings[new_cat] = emb
        group_counter += 1
        return new_cat, best_score

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(layout="centered")
st.title("📂 TMF Reporter 3")
st.write("Enhanced report categorizer.")

# Function to filter messages based on base names
def filter_messages(file_contents, base_names):
    # timestamp_pattern = re.compile(r'\[\d{2}:\d{2}, \d{1,2}/\d{1,2}/\d{4}\]|^\[\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2} [APM]{2}]')
    timestamp_pattern = re.compile(r'\[\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2} (?:am|pm)\]|\[\d{1,2}:\d{2} (?:am|pm), \d{1,2}/\d{1,2}/\d{4}\]|\[\d{1,2}:\d{2}, \d{1,2}/\d{1,2}/\d{4}\]|^\[\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2} [APM]{2}]')
    name_patterns = [
        re.compile(rf'\b{re.escape(name)}\b', re.IGNORECASE) if re.match(r'\w+', name)
        else re.compile(rf'{re.escape(name)}', re.IGNORECASE)  # No word boundary for non-word characters
        for name in base_names
    ]

    filtered_lines = []
    skip_block = False
    current_message = []

    for line in file_contents.splitlines():
        if timestamp_pattern.match(line):
            if current_message:
                filtered_lines.append(' '.join(current_message).strip().lower())
                current_message = []

            if any(pattern.search(line) for pattern in name_patterns):
                skip_block = True
            else:
                skip_block = False

        if not skip_block:
            current_message.append(line.strip().lower())

    if not skip_block and current_message:
        filtered_lines.append(' '.join(current_message).strip().lower())

    return '\n\n'.join(filtered_lines)

# Function to process all files for Process 1
def process_uploaded_files_filtering(uploaded_files, base_names):
    all_output = []

    for uploaded_file in uploaded_files:
        file_contents = uploaded_file.read().decode("utf-8")
        filtered_text = filter_messages(file_contents, base_names)
        all_output.append(f"===Cleansed content from {uploaded_file.name}:===\n{filtered_text}")
    
    combined_output = "\n\n".join(all_output)
    return combined_output

tab1, tab2, tab3 = st.tabs(["Text Cleansing", "Categorizer", "Categorize Single Message"])

# ------------------------
# Tab 1: File categorizer
# ------------------------
with tab1:
    st.subheader("🧽 Text Cleansing")

    base_names_input = st.text_area(
        "Enter names (to be removed when cleansing text file)",
        "Hartina, Tina, Normah, Pom, Afizan, Pijan, Ariff, Arep, Arip, Dheffirdaus, Dhef, Dheff, Dheft, Hazrina, Rina, Nurul, Huda, Zazarida, Zaza, Eliasaph, Wan, ] : , ] :"
    )
    base_names = [name.strip() for name in base_names_input.split(",")]

    uploaded_files_filter = st.file_uploader(
        "Upload text file for cleansing (max 2)", type="txt", accept_multiple_files=True
    )

    if uploaded_files_filter and len(uploaded_files_filter) > 2:
        st.error("You can only upload up to 2 files.")
    else:
        if uploaded_files_filter and st.button('Cleanse file'):
            filtered_output = process_uploaded_files_filtering(uploaded_files_filter, base_names)

            # CSS to disable cursor change
            st.markdown(
                """
                <style>
                .stTextArea textarea[disabled] { cursor: default; }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Display output
            st.text_area("Cleansed Output", value=filtered_output, height=400, disabled=True)

            # Download
            download_data = BytesIO(filtered_output.encode("utf-8"))
            st.download_button(
                label="Download cleansed text",
                data=download_data,
                file_name="cleansed_output.txt",
                mime="text/plain"
            )
with tab2:
    st.subheader("🛠 Categorizer")
    
    uploaded_file = st.file_uploader("Upload cleansed_output.txt", type=["txt"])

    if uploaded_file:
        # Read messages
        lines = uploaded_file.read().decode("utf-8").splitlines()
        messages = []

        for line in lines:
            if "]" in line and ":" in line:
                msg = line.split(":", 2)[-1].strip()
                if msg and has_valid_id(msg):
                    # filter only with valid IDs
                    messages.append(msg)

        st.success(f"Loaded {len(messages)} messages (filtered unnecessary text)")

        # Run categorization
        results = []
        for msg in messages:
            cat, score = categorize_message(msg)
            results.append((msg, cat, round(score, 2)))

        # Choose View Mode
        view_mode = st.radio("Select View Mode", ["Developer View", "User View"])
        st.markdown("---")  # horizontal line

        if view_mode == "Developer View":
            # Display results in scrollable text area
            st.subheader("Categorized Messages (by ID)")
        
            dev_output_lines = []
            expanded_results = []  # store per-ID results
        
            for msg, cat, score in results:
                ids = extract_ids(msg)
                if ids:
                    for tid in ids:
                        dev_output_lines.append(f"[{cat}] ({score:.2f}) → {tid} | {msg}")
                        expanded_results.append((tid, cat, score, msg))
                else:
                    # if no ID, still include the message
                    dev_output_lines.append(f"[{cat}] ({score:.2f}) → {msg}")
                    expanded_results.append(("N/A", cat, score, msg))
        
            dev_output_text = "\n\n".join(dev_output_lines)
        
            # Custom CSS: default cursor in disabled textarea
            st.markdown(
                """
                <style>
                textarea[disabled] {
                    cursor: default !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
        
            st.text_area("Results", dev_output_text, height=500, disabled=True)
            st.markdown("---")
        
            # Summary (count by IDs instead of messages)
            st.subheader("📊 Category Summary")
            summary = {}
            for tid, cat, _, _ in expanded_results:
                summary[cat] = summary.get(cat, 0) + 1
        
            st.table([{"Category": k, "Count": v} for k, v in summary.items()])


        else:
            # User-friendly grouping
            st.subheader("📋 Grouped by Category")
            grouped = {}
            seen_ids = set()  # ensure unique IDs across all categories

            for msg, cat, _ in results:
                ids = extract_ids(msg)
                if not ids:
                    continue
                for tid in ids:
                    if tid in seen_ids:
                        continue
                    seen_ids.add(tid)
                    if cat not in grouped:
                        grouped[cat] = set()
                    grouped[cat].add(tid)

            # Build formatted string
            output_lines = []
            for cat, ids in grouped.items():
                output_lines.append(f"{cat}:")
                for tid in sorted(ids):
                    output_lines.append(f"  {tid}")
                output_lines.append("")  # spacing
            output_text = "\n".join(output_lines)

            # Custom CSS: default cursor in disabled textarea
            st.markdown(
                """
                <style>
                textarea[disabled] {
                    cursor: default !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Display inside scrollable, taller text area
            st.text_area("Grouped Results", output_text, height=500, disabled=True)

            # ------------------------
            # Export to Excel
            # ------------------------
            st.markdown("---")
            export_data = []
            for cat, ids in grouped.items():
                for tid in sorted(ids):
                    export_data.append({"Ticket/ID": tid, "Category": cat})

            df_export = pd.DataFrame(export_data)

            # Generate filename with today's date
            today_str = datetime.today().strftime("%d.%m.%Y")
            filename = f"FF TT Report {today_str}.xlsx"

            # Convert to Excel in memory
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df_export.to_excel(writer, index=False, sheet_name="Report")
            output.seek(0)

            st.subheader("📥 Export Report to Excel")
            st.download_button(
                label="Download Excel Report",
                data=output,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# ------------------------
# Tab 2: Test single message
# ------------------------
with tab3:
    st.subheader("🔎 Categorize Single Message")
    test_msg = st.text_input("Enter a message:")

    if test_msg:
        cat, score = categorize_message(test_msg)
        st.write(f"Prediction: **{cat}** (score={score:.2f})")
