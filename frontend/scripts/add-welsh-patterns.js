'use strict';
/**
 * add-welsh-patterns.js
 * Adds comprehensive Welsh-language patterns to every intent in knowledge.json.
 * Run: node scripts/add-welsh-patterns.js
 * Then: node scripts/build-naivebayes-training.js   (rebuild classifier sidecar)
 */

const fs   = require('fs');
const path = require('path');

const KNOWLEDGE_PATH = path.join(process.cwd(), 'knowledge.json');
const knowledge = JSON.parse(fs.readFileSync(KNOWLEDGE_PATH, 'utf8'));

// ── Welsh patterns keyed by intent tag ────────────────────────────────────────
// Each array contains new Welsh patterns to ADD (deduped against existing ones).
const WELSH_ADDITIONS = {

  // ── Greetings & smalltalk ─────────────────────────────────────────────────
  greeting: [
    'shwmae', 'shwmai', 'sut wyt ti', 'sut mae', 'sut ydych chi',
    'bore da', 'prynhawn da', 'noswaith dda', 'helo', 'helô',
    'heia', 'iawn', 'oes help', 'helo bawb', 'sut hwyl',
    'helo upal', 'shwmae upal', 'sut wyt ti heddiw',
    'ga i ofyn cwestiwn', 'alla i gael help', 'oes rhywun yno',
  ],

  capabilities: [
    'beth allwch chi ei wneud', 'sut allwch chi helpu',
    'beth yw eich pwrpas', 'ydych chi yn bot', 'ydych chi yn AI',
    'allwch chi fy neall', 'beth yw U-Pal', 'pwy ydych chi',
    'pa fathau o gwestiynau allaf ofyn', 'beth allaf ei ofyn i chi',
    'sut wyt ti yn gweithio', 'pa bynciau ydych chi yn eu gwybod',
    'allwch chi helpu gyda materion prifysgol',
    'beth mae U-Pal yn ei wneud', 'sut mae defnyddio U-Pal',
    'pa gymorth sydd ar gael', 'helpwch fi ddeall yr hyn rydych chi yn ei wneud',
  ],

  goodbye: [
    'hwyl', 'hwyl fawr', 'da bo ti', 'da bo chi', 'ta ra',
    'wela i di', 'tan y tro nesaf', 'diolch a hwyl',
    'mae rhaid i mi fynd', 'dwi am adael nawr', 'hwyl am y tro',
    'tan tro nesaf', 'nos da', 'pob hwyl',
  ],

  thanks: [
    'diolch', 'diolch yn fawr', 'diolch yn fawr iawn', 'diolch byth',
    'diolch am eich help', 'diolch am helpu', 'gwych diolch',
    'perffaith diolch', 'ardderchog diolch', 'da iawn diolch',
    'mi wnes i ddeall diolch', 'wedi cael yr ateb diolch',
  ],

  how_are_you: [
    'sut wyt ti', 'sut mae pethau', 'sut hwyl sydd arnot ti',
    'sut ydych chi heddiw', 'wyt ti yn iawn', 'sut mae hi',
    'sut wyt ti yn gwneud', 'popeth yn iawn gyda ti',
    'shwt wyt ti', 'shwd wyt ti',
  ],

  who_are_you: [
    'pwy wyt ti', 'pwy ydych chi', 'beth yw dy enw',
    'beth yw eich enw', 'ai bot wyt ti', 'ai AI yw hon',
    'ydych chi yn gynorthwyydd awtomatig',
  ],

  // ── Admissions ────────────────────────────────────────────────────────────
  admissions_apply: [
    'sut mae gwneud cais i PCYDDS', 'sut ydw i yn ymgeisio',
    'sut mae gwneud cais am gwrs', 'gwneud cais am gwrs PCYDDS',
    'sut i wneud cais drwy UCAS', 'cais UCAS PCYDDS',
    'sut mae ymgeisio i brifysgol', 'ble mae ffurflen gais PCYDDS',
    'proses ymgeisio PCYDDS', 'sut mae cofrestru am gwrs',
    'alla i wneud cais ar lein', 'cyflwyno cais i PCYDDS',
    'sut mae dechrau fy nghais', 'pa ddogfennau sydd eu hangen i wneud cais',
    'dyddiad cau cais UCAS PCYDDS', 'gwneud cais i brifysgol Cymru',
    'sut mae ymgeisio yn uniongyrchol i PCYDDS',
    'sut mae astudio yng Nghymru', 'ffurflen gais ar gyfer israddedig',
  ],

  admissions_requirements: [
    'beth yw gofynion mynediad', 'pa raddau sydd eu hangen',
    'pa gymwysterau sydd eu hangen i astudio', 'lefel A gofynion PCYDDS',
    'a oes angen lefelau A arnaf', 'gofynion mynediad israddedig',
    'pa gyrsiau sydd ar gael heb lefelau A', 'pwynt UCAS gofynnol',
    'a ydw i yn gymwys i ymgeisio', 'gofynion mynediad cwrs nyrsio',
    'sut i wirio a ydw yn gymwys', 'gofynion mynediad ôl-raddedig',
    'oes angen cymhwyster Cymraeg arnaf', 'beth yw gofynion mynediad cyffredinol PCYDDS',
    'pa raddau TGAU sydd eu hangen', 'cymwysterau ar gyfer mynediad',
  ],

  admissions_deadline: [
    'pryd yw dyddiad cau cais UCAS', 'pryd mae ceisiadau yn cau',
    'dyddiad cau ymgeisio i PCYDDS', 'pryd yw dyddiad olaf i wneud cais',
    'sut i wirio dyddiad cau', 'a wnes i golli y dyddiad cau',
    'dyddiad cau mynediad mis Medi', 'pryd mae ceisiadau Clirio yn dechrau',
    'dyddiad cau cais uniongyrchol PCYDDS', 'a allaf wneud cais ar ôl y dyddiad cau',
    'pryd yw dyddiad cau ceisiadau ôl-raddedig',
    'pa mor hir mae ceisiadau yn cymryd', 'pryd i wneud cais am le',
  ],

  admissions_international: [
    'sut mae myfyrwyr rhyngwladol yn ymgeisio', 'ymgeisio o dramor',
    'gofynion fisa myfyriwr', 'sut i gael fisa myfyriwr y DU',
    'ymgeisio o wlad arall', 'a yw PCYDDS yn derbyn myfyrwyr rhyngwladol',
    'mynediad myfyrwyr y tu allan i r DU', 'gofynion IELTS PCYDDS',
    'cymorth i fyfyrwyr rhyngwladol', 'gwybodaeth am fisa myfyriwr',
    'sut mae dod i astudio yng Nghymru o dramor',
    'pa raddau Saesneg sydd eu hangen', 'ysgoloriaethau i fyfyrwyr rhyngwladol',
  ],

  admissions_foundation: [
    'beth yw cwrs sylfaen', 'cwrs sylfaen PCYDDS',
    'a oes gan PCYDDS flwyddyn sylfaen', 'sut mae mynediad drwy lwybr sylfaen',
    'a allaf wneud blwyddyn sylfaen cyn fy ngradd',
    'pa gyrsiau mynediad sydd ar gael', 'mynediad amgen i brifysgol',
    'cwrs mynediad PCYDDS', 'sut mae astudio heb lefelau A drwy gwrs sylfaen',
    'a yw r flwyddyn sylfaen yn cyfri tuag at fy ngradd',
  ],

  // ── Courses ───────────────────────────────────────────────────────────────
  courses_general: [
    'pa gyrsiau sydd ar gael yn PCYDDS', 'beth yw r cyrsiau sydd ar gael',
    'pa raddau sydd ar gael', 'rhestr o gyrsiau PCYDDS',
    'pa bynciau mae PCYDDS yn eu cynnig', 'sut mae dod o hyd i gwrs',
    'pa raddau sylfaen sydd ar gael', 'pa fodiwlau sydd ar y cwrs',
    'pa gyrsiau ôl-raddedig sydd ar gael yn PCYDDS',
    'a yw PCYDDS yn cynnig cyrsiau ar lein', 'sut mae chwilio am gyrsiau',
    'cyrsiau rhan amser PCYDDS', 'cyrsiau amser llawn PCYDDS',
    'sut i ddod o hyd i gwrs sy n addas i mi', 'pa raddau ymchwil sydd ar gael',
    'cyrsiau nos PCYDDS', 'astudio yn PCYDDS',
  ],

  courses_computing: [
    'a oes gan PCYDDS gwrs cyfrifiadureg', 'gradd cyfrifiadureg PCYDDS',
    'cwrs TG PCYDDS', 'gradd gwyddoniaeth cyfrifiadurol',
    'cyrsiau technoleg gwybodaeth', 'a yw PCYDDS yn cynnig BSc Cyfrifiadureg',
    'gradd meddalwedd PCYDDS', 'cwrs peirianneg meddalwedd',
    'graddau seiberddiogelwch PCYDDS', 'cyfrifiaduron a systemau PCYDDS',
    'cwrs data science', 'gradd AI dysgu peiriant PCYDDS',
  ],

  courses_nursing: [
    'a oes gradd nyrsio yn PCYDDS', 'cwrs nyrsio PCYDDS',
    'sut mae gwneud cais am gwrs nyrsio', 'hyfforddiant nyrs PCYDDS',
    'gradd bydwreigiaeth PCYDDS', 'cwrs gwyddor iechyd PCYDDS',
    'nyrsio a gofal iechyd PCYDDS', 'sut i fod yn nyrs drwy PCYDDS',
    'a yw PCYDDS yn cynnig nyrsio cymunedol', 'gradd iechyd meddwl PCYDDS',
    'ymgeisio am gwrs nyrsio', 'nyrsio plant PCYDDS',
  ],

  courses_education: [
    'a oes cwrs addysg yn PCYDDS', 'gradd addysg gynradd PCYDDS',
    'TAR PCYDDS', 'cwrs hyfforddi athrawon PCYDDS',
    'a yw PCYDDS yn cynnig TAR uwchradd', 'sut i hyfforddi fel athro yn PCYDDS',
    'cwrs addysg blynyddoedd cynnar PCYDDS', 'gradd addysg arbennig',
    'hyfforddiant addysg PCYDDS', 'sut mae ymgeisio am gwrs TAR',
    'cyrsiau addysgu Cymraeg', 'ymgeisio am gwrs athro',
  ],

  // ── Fees & Finance ────────────────────────────────────────────────────────
  fees_tuition: [
    'faint yw r ffioedd dysgu', 'faint mae n costio i astudio yn PCYDDS',
    'ffioedd myfyrwyr cartref', 'ffioedd dysgu ar gyfer 2025 26',
    'faint yw ffioedd israddedig', 'faint yw r gost y flwyddyn',
    'a oes rhaid talu ffioedd o flaen llaw', 'sut mae talu ffioedd dysgu',
    'ffioedd cwrs PCYDDS', 'pa mor ddrud yw astudio yn PCYDDS',
    'ffioedd rhan amser PCYDDS', 'faint yw ffioedd ôl raddedig',
    'faint yw r ffioedd PhD', 'ffioedd TAR PCYDDS',
    'costau astudio yn PCYDDS', 'ffioedd blynyddol PCYDDS',
  ],

  fees_student_loan: [
    'sut mae gwneud cais am gyllid myfyrwyr', 'benthyciad myfyrwyr Cymru',
    'sut mae cael benthyciad cynhaliaeth', 'Cyllid Myfyrwyr Cymru sut i wneud cais',
    'a oes benthyciad ffioedd dysgu ar gael', 'a allaf gael benthyciad i astudio',
    'faint o fenthyciad alla i ei gael', 'benthyciad myfyrwyr ar gyfer ôl raddedig',
    'sut mae ad dalu benthyciad myfyrwyr', 'cyllid ar gyfer myfyrwyr o Gymru',
    'pryd mae arian myfyrwyr yn cael ei dalu', 'pa gymorth ariannol sydd ar gael',
    'a allaf ymgeisio am fenthyciad cynhaliaeth', 'cymorth ariannol i fyfyrwyr Cymru',
  ],

  fees_scholarships: [
    'a oes ysgoloriaethau ar gael yn PCYDDS', 'bwrsarïau PCYDDS',
    'sut mae ymgeisio am ysgoloriaeth', 'cymorth ariannol PCYDDS',
    'a oes bwrsari ar gael i fyfyrwyr cartref', 'ysgoloriaethau i fyfyrwyr rhyngwladol PCYDDS',
    'sut mae cael bwrsari', 'pa ysgoloriaethau sydd ar gael',
    'a yw PCYDDS yn cynnig grantiau', 'cymorth ariannol ychwanegol i fyfyrwyr',
    'ysgoloriaethau ar gyfer myfyrwyr cymraeg eu hiaith', 'bwrsari cymraeg PCYDDS',
    'ariannu ar gyfer PhD PCYDDS', 'ysgoloriaethau ymchwil PCYDDS',
  ],

  fees_and_finance: [
    'costau byw fel myfyriwr', 'sut i gyllido astudio', 'beth yw r ffioedd ac ariannu',
    'pa gymorth ariannol sydd ar gael i fyfyrwyr', 'sut mae rheoli arian fel myfyriwr',
    'costau byw yn Abertawe fel myfyriwr', 'a allaf fforddio astudio yn PCYDDS',
    'arian myfyrwyr PCYDDS', 'ariannu israddedig', 'ariannu ôl raddedig',
    'cymorth gyda chost astudio', 'sut mae talu am fy astudiaethau',
  ],

  financial_support: [
    'cronfa caledi PCYDDS', 'oes cronfa caledi ar gael', 'help ariannol brys',
    'dw i mewn trafferth ariannol', 'alla i ddim fforddio rhent',
    'sut mae gwneud cais i gronfa galedi', 'pa gymorth ariannol brys sydd ar gael',
    'cymorth gyda chostau byw', 'arian brys i fyfyrwyr', 'grantiau caledi PCYDDS',
    'alla i ddim fforddio bwyd', 'dw i angen cymorth ariannol ar unwaith',
    'sut i gael cymorth ariannol brys', 'cronfa argyfwng myfyrwyr PCYDDS',
    'help os yw arian yn broblem', 'oes grant ar gael os ydw i mewn angen',
  ],

  // ── IT & Systems ──────────────────────────────────────────────────────────
  it_portal: [
    'sut mae mewngofnodi i MyTSD', 'porth myfyrwyr PCYDDS',
    'alla i ddim cael mynediad i MyTSD', 'sut mae defnyddio MyTSD',
    'beth yw MyTSD', 'sut mae gweld fy amserlen ar MyTSD',
    'cyfeiriad MyTSD', 'sut mae newid fy nghyfrinair ar MyTSD',
    'MyTSD ar gyfer enrolment', 'sut mae cofrestru ar MyTSD',
    'pam alla i ddim mewngofnodi i r porth myfyrwyr',
    'sut i gyrchu MyTSD', 'system myfyrwyr PCYDDS',
  ],

  it_moodle: [
    'sut mae defnyddio Moodle', 'alla i ddim cael mynediad i Moodle',
    'cyfeiriad Moodle PCYDDS', 'sut mae mewngofnodi i Moodle',
    'pam alla i ddim gweld fy modiwlau ar Moodle', 'Moodle PCYDDS',
    'sut mae lawrlwytho deunyddiau cwrs o Moodle',
    'sut mae cyflwyno aseiniad ar Moodle', 'VLE PCYDDS',
    'amgylchedd dysgu rhithwir PCYDDS', 'pwy i gysylltu â nhw am broblemau Moodle',
    'sut mae gweld fy ngraddau ar Moodle', 'cymorth Moodle PCYDDS',
  ],

  it_helpdesk: [
    'sut mae cysylltu â chymorth TG', 'cymorth TG PCYDDS',
    'problem gyda chyfrifiadur', 'alla i ddim mewngofnodi i r system',
    'ailosod cyfrinair PCYDDS', 'sut mae ailosod fy nghyfrinair',
    'cymorth TG ar gyfer myfyrwyr', 'desg gymorth TG PCYDDS',
    'problem dechnegol yn PCYDDS', 'sut i gael cymorth gyda systemau TG',
    'pwy i gysylltu â nhw am broblemau TG', 'e bost TG PCYDDS',
    'rhif ffôn cymorth TG', 'oriau cymorth TG PCYDDS',
    'mae r system wedi torri sut mae cael help',
  ],

  it_wifi: [
    'sut mae cysylltu â wifi ar gampws', 'wifi ar gampws PCYDDS',
    'cyfrinair wifi PCYDDS', 'rhwydwaith eduroam PCYDDS',
    'alla i ddim cysylltu â r rhyngrwyd', 'wifi myfyrwyr PCYDDS',
    'sut mae defnyddio eduroam', 'rhwydwaith di wifr PCYDDS',
    'pa rwydwaith wifi sydd ar y campws', 'problemau wifi ar y campws',
    'alla i ddefnyddio wifi ar bob campws', 'sut i gael mynediad i r rhyngrwyd yn PCYDDS',
  ],

  uwtsd_digital_systems: [
    'pa systemau digidol mae PCYDDS yn eu defnyddio', 'systemau ar lein PCYDDS',
    'sut mae defnyddio systemau ar lein y brifysgol',
    'cyfrifon digidol PCYDDS myfyrwyr', 'sut mae cael mynediad i bopeth ar lein',
    'pa apiau a phlatfformau mae PCYDDS yn eu defnyddio',
    'Microsoft 365 PCYDDS', 'cyfrif e bost myfyriwr PCYDDS',
    'sut mae sefydlu fy nghyfrif myfyriwr', 'systemau rhithwir PCYDDS',
    'sut mae cael mynediad i r holl blatfformau myfyrwyr',
    'beth yw systemau TG myfyrwyr PCYDDS', 'sut mae dod o hyd i fanylion mewngofnodi',
  ],

  // ── Accommodation ─────────────────────────────────────────────────────────
  accommodation_general: [
    'sut mae gwneud cais am lety', 'llety myfyrwyr PCYDDS',
    'a oes neuaddau preswyl ar gael', 'ble alla i fyw fel myfyriwr yn PCYDDS',
    'sut mae dod o hyd i lety', 'cais llety myfyrwyr Abertawe',
    'neuaddau preswyl PCYDDS', 'porth HallPad PCYDDS',
    'a yw llety wedi ei warantu i fyfyrwyr blwyddyn gyntaf',
    'sut mae byw ar gampws', 'mathau o lety PCYDDS',
    'a oes llety hunangynhwysol ar gael', 'llety sengl PCYDDS',
    'llety rhannu PCYDDS', 'oriau cymorth llety',
    'cymorth llety PCYDDS', 'sut mae newid llety',
    'sut mae canslo llety', 'rheolau llety PCYDDS',
    'pa mor gynnar ddylwn i wneud cais am lety',
    'dyddiad cau ceisiadau llety',
  ],

  accommodation_cost: [
    'faint mae llety yn ei gostio yn PCYDDS', 'costau neuaddau preswyl',
    'faint yw r rhent mewn neuaddau', 'costau wythnosol llety PCYDDS',
    'faint yw r gost am ystafell sengl', 'pa mor ddrud yw r neuaddau preswyl',
    'cymharu costau llety PCYDDS', 'oes dewis o lety rhatach ar gael',
    'costau cyfleustodau yn y neuaddau', 'a yw r biliau wedi eu cynnwys yn y rhent',
    'faint yw r blaendal llety', 'sut i dalu am lety mewn rhandaliadau',
  ],

  // ── Campuses ─────────────────────────────────────────────────────────────
  campuses_general: [
    'pa gampysau sydd gan PCYDDS', 'ble mae campysau PCYDDS',
    'sawl campws sydd gan PCYDDS', 'sut mae teithio rhwng campysau PCYDDS',
    'pa gampws ddylwn i astudio arno', 'cyfleusterau campws PCYDDS',
    'pa gampws sydd orau ar gyfer fy nghwrs',
    'sut mae cyrraedd campysau PCYDDS', 'gwybodaeth am gampysau PCYDDS',
    'a oes cegin ar y campws', 'cyfleusterau y campws',
    'pa wasanaethau sydd ar y campws', 'bwyd a diod ar y campws',
    'oriau agor adeilad y campws', 'sut mae archebu ystafell ar y campws',
  ],

  campuses_swansea: [
    'ble mae campws Abertawe', 'cyfeiriad campws SA1',
    'sut mae cyrraedd campws SA1', 'campws SA1 Abertawe',
    'campws Mount Pleasant Abertawe', 'campws Abertawe PCYDDS',
    'pa gampysau sydd yn Abertawe', 'cyfleusterau campws SA1',
    'sut mae cyrraedd Abertawe ar drên', 'parcio yn campws SA1',
    'sut mae cyrraedd Mount Pleasant PCYDDS',
    'cyfeiriad SA1 Glannau Abertawe', 'adeilad IQ Abertawe PCYDDS',
  ],

  campuses_carmarthen: [
    'ble mae campws Caerfyrddin', 'cyfeiriad campws Caerfyrddin PCYDDS',
    'sut mae cyrraedd campws Caerfyrddin', 'campws Caerfyrddin PCYDDS',
    'cyfleusterau campws Caerfyrddin', 'parcio yn Caerfyrddin PCYDDS',
    'sut mae cyrraedd Caerfyrddin ar fws', 'oriau campws Caerfyrddin',
    'beth sydd ar gampws Caerfyrddin', 'pa gyrsiau sydd yng Nghaerfyrddin',
  ],

  campuses_lampeter: [
    'ble mae campws Llambed', 'cyfeiriad campws Llambed PCYDDS',
    'sut mae cyrraedd campws Llambed', 'campws Llambed PCYDDS',
    'cyfleusterau campws Llambed', 'sut mae cyrraedd Llambed ar fws',
    'pa gyrsiau sydd yn Llambed', 'Llyfrgell y Sylfaenwyr Llambed',
    'oriau campws Llambed', 'beth sydd ar gampws Llambed',
    'sut mae cyrraedd Llambed o Abertawe',
  ],

  campus_locations: [
    'ble mae PCYDDS', 'cyfeiriad prifysgol Cymru Y Drindod Dewi Sant',
    'sut mae dod o hyd i gampws PCYDDS', 'sut mae cyrraedd PCYDDS',
    'pa mor bell yw r campws', 'cyfeiriad PCYDDS', 'map campws PCYDDS',
    'parcio ar gampws PCYDDS', 'cyfleusterau maes parcio',
    'sut mae mynd i r campws ar drafnidiaeth gyhoeddus',
  ],

  // ── Library ───────────────────────────────────────────────────────────────
  library: [
    'oriau agor y llyfrgell', 'pryd mae r llyfrgell ar agor',
    'sut mae benthyca llyfr o r llyfrgell', 'sut mae defnyddio r llyfrgell',
    'cyfleusterau llyfrgell PCYDDS', 'mannau astudio yn y llyfrgell',
    'ble mae r llyfrgell', 'cymorth llyfrgell PCYDDS',
    'sut mae cael mynediad i adnoddau ar lein', 'databas ar lein PCYDDS',
    'a allaf argraffu yn y llyfrgell', 'cyfnodolion ar lein PCYDDS',
    'sut mae adnewyddu llyfrau', 'dirwy llyfrau hwyr',
    'pa adnoddau sydd ar gael yn y llyfrgell',
    'oriau llyfrgell yn ystod gwyliau', 'staff llyfrgell PCYDDS',
    'sut i gysylltu â r llyfrgell', 'ystafell dawel astudio',
  ],

  library_swansea: [
    'oriau llyfrgell Abertawe', 'pryd mae llyfrgell SA1 ar agor',
    'ble mae llyfrgell SA1 PCYDDS', 'cyfleusterau llyfrgell Abertawe PCYDDS',
    'sut mae cyrraedd llyfrgell SA1', 'llyfrgell campws Abertawe PCYDDS',
  ],

  library_carmarthen: [
    'oriau llyfrgell Caerfyrddin', 'pryd mae llyfrgell Caerfyrddin ar agor',
    'ble mae llyfrgell Caerfyrddin PCYDDS', 'cyfleusterau llyfrgell Caerfyrddin',
    'sut mae cyrraedd llyfrgell Caerfyrddin', 'llyfrgell campws Caerfyrddin',
  ],

  library_lampeter: [
    'oriau llyfrgell Llambed', 'pryd mae llyfrgell Llambed ar agor',
    'ble mae llyfrgell Llambed PCYDDS', 'Llyfrgell y Sylfaenwyr Llambed',
    'cyfleusterau llyfrgell Llambed', 'sut mae cyrraedd llyfrgell Llambed',
  ],

  // ── Wellbeing ─────────────────────────────────────────────────────────────
  wellbeing_general: [
    'sut mae cael cymorth lles', 'gwasanaeth lles PCYDDS',
    'dw i angen cymorth iechyd meddwl', 'cymorth cwnsela PCYDDS',
    'dw i dan straen yn y brifysgol', 'dw i n teimlo yn bryderus am arholiadau',
    'dw i n teimlo wedi fy llethu gan waith prifysgol',
    'yn poeni am fy aseiniad', 'dw i n cael hi n anodd yn y brifysgol',
    'angen siarad â rhywun am fy iechyd meddwl',
    'cymorth lles myfyrwyr PCYDDS', 'sut mae trefnu apwyntiad lles',
    'lles a chymorth PCYDDS', 'ble mae r hwb lles',
    'sut mae cysylltu â r tîm lles', 'cymorth emosiynol i fyfyrwyr',
    'dw i n teimlo yn isel fy ysbryd', 'dw i n cael trafferth ymdopi',
    'cymorth iechyd meddwl am ddim i fyfyrwyr', 'cwnsela am ddim PCYDDS',
    'dw i n teimlo ar goll yn y brifysgol', 'straen traethawd hir cymorth',
    'pryder arholiad cymorth PCYDDS', 'yn poeni am beidio â phasio',
  ],

  wellbeing_crisis: [
    'dw i n teimlo n anobeithiol ac unig', 'meddyliau hunanladdol',
    'dw i eisiau lladd fy hun', 'lladd fy hun heno',
    'yn meddwl am roi terfyn ar fy mywyd', 'hunan niweidio',
    'eisiau brifo fy hun', 'alla i ddim ymdopi mwy',
    'argyfwng iechyd meddwl ar unwaith', 'anobeithiol iawn methu ymdopi',
    'does dim rheswm i fyw', 'yn meddwl am hunan laddiad',
    'yn colli fy meddwl', 'mae popeth yn rhy anodd',
  ],

  wellbeing_disability: [
    'cymorth anabledd PCYDDS', 'a oes cymorth anabledd ar gael',
    'sut mae cael cymorth DSA', 'cymorth dyslecsia PCYDDS',
    'anawsterau dysgu PCYDDS', 'sut mae cofrestru gyda r gwasanaeth anabledd',
    'a allaf gael amser ychwanegol mewn arholiadau', 'cymorth mynediad PCYDDS',
    'cymorth ar gyfer myfyrwyr ag anableddau', 'ALSS PCYDDS',
    'cymorth dysgu ychwanegol', 'cymorth ar gyfer awtistiaeth',
    'cymorth ar gyfer anhwylder pryder', 'a oes cymhorthion arbennig ar gael',
    'sut mae gwneud cais am gymorth anabledd', 'DSA PCYDDS sut i wneud cais',
  ],

  // ── Student Services ──────────────────────────────────────────────────────
  students_union: [
    'undeb y myfyrwyr PCYDDS', 'beth mae undeb y myfyrwyr yn ei wneud',
    'sut mae ymuno ag undeb y myfyrwyr', 'clybiau a chymdeithasau PCYDDS',
    'pa gymdeithasau sydd ar gael yn PCYDDS', 'sut mae sefydlu cymdeithas newydd',
    'digwyddiadau undeb myfyrwyr PCYDDS', 'swyddfa undeb y myfyrwyr',
    'cymorth cymheiriaid PCYDDS', 'pa glybiau chwaraeon sydd ar gael',
    'gweithgareddau myfyrwyr PCYDDS', 'rhaglenni gwirfoddoli PCYDDS',
    'sut mae cymryd rhan yn y brifysgol', 'sut mae etholio am swydd yn yr undeb',
  ],

  student_services: [
    'gwasanaethau myfyrwyr PCYDDS', 'sut mae cysylltu â gwasanaethau myfyrwyr',
    'ble mae gwasanaethau myfyrwyr', 'pa gymorth sydd ar gael i fyfyrwyr',
    'cymorth cyffredinol i fyfyrwyr PCYDDS', 'oriau gwasanaethau myfyrwyr',
    'sut i gysylltu â r tîm cymorth', 'sut mae cael cymorth gan y brifysgol',
    'pa wasanaethau sydd ar gael i fyfyrwyr PCYDDS', 'gwybodaeth am wasanaethau myfyrwyr',
    'ble i fynd os oes problem gyda r astudiaethau', 'pa adnoddau cymorth sydd ar gael',
  ],

  human_agent: [
    'siarad â rhywun go iawn', 'cysylltu ag aelod o staff',
    'sut mae siarad â pherson', 'dw i eisiau siarad â rhywun byw',
    'alla i siarad â dynol', 'sut mae cael siarad â staff',
    'trosglwyddo i asiant', 'cysylltu â pherson',
    'asiant byw PCYDDS', 'siarad ag asiant cymorth',
    'dw i angen siarad â pherson go iawn', 'cysylltu â staff PCYDDS',
    'pa mor hir yw r amser aros i siarad â rhywun',
  ],

  contact_general: [
    'sut mae cysylltu â PCYDDS', 'manylion cyswllt PCYDDS',
    'rhif ffôn PCYDDS', 'cyfeiriad e bost PCYDDS',
    'sut i gysylltu â r brifysgol', 'e bost cyffredinol PCYDDS',
    'cyfeiriad PCYDDS', 'pa mor hir yw r amser ymateb',
    'oriau agor gweinyddiaeth PCYDDS', 'sut mae cysylltu â r adran',
    'manylion cyswllt ar gyfer ymholiadau', 'sut mae ffonio PCYDDS',
    'derbynfa PCYDDS', 'e bost ymholiadau PCYDDS',
  ],

  // ── Academic ──────────────────────────────────────────────────────────────
  graduation: [
    'pryd mae r seremoni raddio', 'sut mae cofrestru am raddio',
    'sut mae llogi gŵn graddio', 'tocynnau ar gyfer y seremoni raddio',
    'faint o westeion alla i ddod â nhw i r graddio',
    'rwy n graddio eleni sut mae trefnu hynny',
    'dyddiadau seremoni raddio PCYDDS', 'lleoliad y seremoni raddio',
    'sut mae archebu ar gyfer graddio', 'gwybodaeth am raddio PCYDDS',
    'sut mae cael fy nhystysgrif gradd', 'trefniadau graddio PCYDDS',
    'llogi gŵn graddio PCYDDS', 'rhaglen y diwrnod graddio',
  ],

  results_grades: [
    'sut mae gweld fy marciau', 'pryd mae r canlyniadau yn cael eu cyhoeddi',
    'sut mae gwirio fy ngraddau', 'ble mae fy marciau ar gael',
    'sut mae cael fy nghanlyniadau arholiad', 'pryd byddaf yn cael fy marciau',
    'sut mae deall fy nhrawsgrifiad', 'a all fy marciau gael eu hadolygu',
    'sut mae apelio yn erbyn canlyniad', 'canlyniadau modiwl PCYDDS',
    'sut mae gweld fy nghanlyniad terfynol', 'ble mae fy nhrawsgrifiad academaidd',
  ],

  timetable: [
    'sut mae gweld fy amserlen', 'ble mae fy amserlen ar gael',
    'pryd mae fy narlithoedd', 'sut mae dod o hyd i fy nhrefn ddyddiol',
    'amserlen myfyrwyr PCYDDS', 'sut mae gwirio fy amserlen dosbarth',
    'a yw fy amserlen ar MyTSD', 'pryd mae fy nhiwtorialau',
    'sut mae gwybod pryd mae fy nosbarth nesaf', 'amserlen ar gyfer y tymor',
    'pryd mae r wythnos gyntaf o ddarlithoedd', 'sut i lawrlwytho fy amserlen',
    'amserlen arholiad PCYDDS', 'sut mae cael fy amserlen arholiad',
  ],

  enrolment_tasks: [
    'sut mae cofrestru yn PCYDDS', 'tasgau cofrestru PCYDDS',
    'sut mae cwblhau cofrestru', 'pryd mae r cofrestru yn cychwyn',
    'sut mae dewis fy modiwlau', 'cyfnod cofrestru PCYDDS',
    'cofrestru am y flwyddyn academaidd newydd', 'ailgofrestru PCYDDS',
    'sut mae cofrestru fel myfyriwr newydd', 'problem gyda chlofrestru ar lein',
    'tasgau sydd eu hangen i gwblhau cofrestru', 'porth cofrestru myfyrwyr',
    'cofrestru ar gyfer modiwlau PCYDDS', 'sut mae cadarnhau fy nghwrs',
  ],

  printing: [
    'sut mae argraffu ar y campws', 'cyfleusterau argraffu PCYDDS',
    'ble mae r argraffwyr ar y campws', 'sut mae defnyddio r argraffwr',
    'credydau argraffu PCYDDS', 'sut mae ychwanegu credydau argraffu',
    'argraffu yn y llyfrgell', 'cost argraffu yn PCYDDS',
    'a allaf argraffu yn lliw ar y campws', 'pa argraffwyr sydd ar gael',
    'sut mae sganu dogfen yn PCYDDS', 'copïo yn PCYDDS',
  ],

  student_stress: [
    'dw i dan straen mawr', 'mae r brifysgol yn rhy anodd i mi',
    'dw i n teimlo dan bwysau mawr', 'straen academaidd PCYDDS cymorth',
    'dw i n teimlo ar goll ac yn ansicr', 'dw i n poeni am fethu fy aseiniad',
    'cymorth pryder arholiad', 'dw i n cael trafferth ymdopi â r llwyth gwaith',
    'dw i n cael hi n anodd gyda r darlithoedd', 'sut mae ymdopi â straen y brifysgol',
    'pryder cyn arholiad', 'yn poeni am beidio â llwyddo',
    'sut mae ymdopi â phwysau r brifysgol', 'cymorth ar gyfer straen myfyrwyr',
    'mae gormod o waith gen i', 'dw i n teimlo n llethu',
  ],

  // ── Language Support ──────────────────────────────────────────────────────
  english_language_support: [
    'cymorth iaith Saesneg PCYDDS', 'a oes cymorth Saesneg ar gael',
    'sut mae gwella fy Saesneg', 'EAP PCYDDS',
    'a oes dosbarthiadau Saesneg am ddim ar gael', 'cymorth ysgrifennu yn Saesneg',
    'cymorth iaith ar gyfer myfyrwyr rhyngwladol', 'IELTS PCYDDS cymorth',
    'sut mae cyfathrebu yn well yn Saesneg', 'cymorth sillafu a gramadeg',
  ],

  welsh_language: [
    'sut mae dysgu Cymraeg yn PCYDDS', 'cyrsiau Cymraeg PCYDDS',
    'a oes cymorth Cymraeg ar gael', 'astudio drwy gyfrwng y Gymraeg',
    'pa gyrsiau sydd ar gael yn Gymraeg', 'cyrsiau Cymraeg i oedolion',
    'sut mae gwella fy Gymraeg', 'cymorth iaith Gymraeg PCYDDS',
    'a yw r cwrs ar gael yn Gymraeg', 'Cymraeg fel ail iaith PCYDDS',
    'hyfforddiant Cymraeg i staff a myfyrwyr', 'dysgu Cymraeg yn gyflym',
  ],

  // ── Careers & Postgrad ────────────────────────────────────────────────────
  career_services: [
    'cymorth gyrfaoedd PCYDDS', 'sut mae cael cyngor gyrfa',
    'gwasanaethau gyrfaoedd PCYDDS', 'sut mae dod o hyd i swydd ar ôl graddio',
    'a oes cymorth ar gyfer cyfweliad', 'sut mae ysgrifennu CV da',
    'interniaethu a lleoliad gwaith PCYDDS', 'pa swyddi sydd ar gael i raddedigion PCYDDS',
    'cymorth chwilio am waith', 'digwyddiadau recriwtio PCYDDS',
    'sut mae dod o hyd i leoliad gwaith', 'cymorth cyflogadwyedd PCYDDS',
    'pa sgiliau sydd eu hangen i gael swydd dda', 'cymorth menter PCYDDS',
  ],

  postgraduate_courses: [
    'pa gyrsiau ôl raddedig sydd ar gael yn PCYDDS',
    'gradd meistr yn PCYDDS', 'MSc PCYDDS',
    'sut mae gwneud cais am gwrs ôl raddedig', 'PhD PCYDDS',
    'a oes ariannu ar gael ar gyfer ôl raddedigion', 'MBA PCYDDS',
    'sut mae astudio ôl raddedig yn PCYDDS', 'PhD rhan amser PCYDDS',
    'cyrsiau ôl raddedig ar lein PCYDDS', 'meistr mewn addysg PCYDDS',
    'graddau ymchwil PCYDDS', 'ôl raddedig a addysgir PCYDDS',
  ],

  undergraduate_courses: [
    'pa raddau israddedig sydd ar gael', 'graddau BSc PCYDDS',
    'gradd BA PCYDDS', 'graddau sylfaen PCYDDS',
    'sut mae dewis fy ngradd', 'pa gwrs israddedig sydd orau i mi',
    'graddau tair blynedd PCYDDS', 'pa bynciau sydd ar gael',
    'gradd israddedig ar lein PCYDDS', 'gradd rhan amser PCYDDS',
  ],

  how_to_apply: [
    'sut mae gwneud cais', 'proses ymgeisio PCYDDS',
    'sut mae dechrau ymgeisio', 'UCAS PCYDDS sut mae gweithio',
    'sut mae cyflwyno cais i PCYDDS', 'sut mae ymgeisio yn uniongyrchol',
    'pa wybodaeth sydd ei hangen ar gyfer y cais', 'camau ymgeisio PCYDDS',
    'sut mae derbyn cyngor am ymgeisio', 'cymorth gyda r broses ymgeisio',
  ],

  uwtsd_about: [
    'beth yw PCYDDS', 'hanes Prifysgol Cymru Y Drindod Dewi Sant',
    'pryd sefydlwyd PCYDDS', 'am PCYDDS', 'beth yw llawn enw PCYDDS',
    'pa mor fawr yw PCYDDS', 'ble mae PCYDDS wedi ei lleoli',
    'a yw PCYDDS yn brifysgol Gymraeg ei hiaith', 'rhengoedd PCYDDS',
    'pa wobrau sydd gan PCYDDS', 'pwy yw arweinyddiaeth PCYDDS',
    'beth mae PCYDDS yn enwog amdano', 'hanes Llambed 1822',
  ],

  // ── Dissertation & Academic Support (new intents) ─────────────────────────
  dissertation_support: [
    'angen cymorth gyda fy nhraethawd hir', 'cymorth traethawd hir PCYDDS',
    'sut mae cysylltu â fy ngoruchwyliwr traethawd hir',
    'sut mae cyflwyno traethawd hir', 'dyddiad cau traethawd hir PCYDDS',
    'a allaf gael estyniad am draethawd hir', 'cymorth ysgrifennu traethawd hir',
    'sut mae strwythuro traethawd hir', 'fformat traethawd hir PCYDDS',
    'dw i n poeni am fy nhraethawd hir', 'cymorth goruchwyliwr traethawd hir',
    'sut mae dod o hyd i fy ngoruchwyliwr', 'pryder traethawd hir PCYDDS',
    'yn cael trafferth gyda fy nhraethawd', 'ymchwil ar gyfer traethawd hir',
  ],

  academic_support: [
    'cymorth academaidd PCYDDS', 'angen cymorth gyda fy astudiaethau',
    'dw i n cwympo y tu ôl â gwaith cwrs', 'pwy all fy helpu gyda fy ngradd',
    'arweiniad academaidd PCYDDS', 'dw i eisiau siarad â thiwtoriaid am fy ngwaith',
    'a allaf siarad ag aelod o staff am fy aseiniad',
    'angen adborth ar fy ngwaith', 'dw i eisiau i rywun wirio fy mhrosiect',
    'a all darlithydd edrych ar fy ngwaith', 'sut mae cael cymorth gyda fy ngradd',
    'ALSS PCYDDS cymorth', 'cymorth sgiliau astudio PCYDDS',
    'dw i n cael trafferth gyda deunydd y cwrs',
    'sut mae cael cymorth ychwanegol', 'cymorth dysgu ychwanegol PCYDDS',
  ],

  staff_contact_query: [
    'sut mae siarad ag aelod o staff', 'dw i eisiau cwrdd â thiwtoriaid',
    'sut mae trefnu apwyntiad gyda thiwtor', 'cymorth un i un PCYDDS',
    'sut mae cysylltu â staff y brifysgol', 'dw i eisiau siarad â rhywun wyneb yn wyneb',
    'a allaf gwrdd â rhywun am fy mhroblem', 'pwy ddylwn i gysylltu â nhw',
    'sut mae cael cymorth personol', 'cymorth personol PCYDDS',
  ],

};

// ── Merge Welsh additions into knowledge base ─────────────────────────────────
let totalAdded = 0;
const updated = knowledge.map(intent => {
  const additions = WELSH_ADDITIONS[intent.tag];
  if (!additions || additions.length === 0) return intent;

  const existingLower = new Set(intent.patterns.map(p => p.toLowerCase().trim()));
  const newPatterns = additions.filter(p => !existingLower.has(p.toLowerCase().trim()));

  if (newPatterns.length === 0) return intent;
  totalAdded += newPatterns.length;
  console.log(`  +${String(newPatterns.length).padStart(2)} Welsh patterns → ${intent.tag}`);
  return { ...intent, patterns: [...intent.patterns, ...newPatterns] };
});

console.log(`\nTotal Welsh patterns added: ${totalAdded}`);
console.log(`Total intents updated: ${updated.filter((u,i) => u !== knowledge[i]).length}`);

// ── Write updated knowledge.json ─────────────────────────────────────────────
fs.writeFileSync(KNOWLEDGE_PATH, JSON.stringify(updated, null, 2), 'utf8');
console.log(`\nSaved to ${KNOWLEDGE_PATH}`);
console.log('Next: node scripts/build-naivebayes-training.js');
