# =====================================================================================================================================
#                                      This File Contains Python Dictionaries With Translations For Out Dataset 
# =====================================================================================================================================


# ========================================================================
#                               COLUMN NAMES
# ========================================================================
column_names_translation = {'Подразделение' : 'Shop' ,
                            'Код' : 'vendor_code' ,
                            'Период, год.Год' : 'sale_year' ,
                            'Период, месяц.Название месяца' : 'sale_month' ,
                            'Период, день.День' : 'sale_day' ,
                            'Заказ клиента / Реализация.Дата.Год' : 'order_year' ,
                            'Заказ клиента / Реализация.Дата.Название месяца' : 'order_month' ,
                            'Заказ клиента / Реализация.Дата.День' : 'order_day' ,
                            'Заказ клиента / Реализация.Дата.Час' : 'order_hour' ,
                            'Заказ клиента / Реализация.Дата.Минута' : 'order_minute' ,
                            'Արտադրման երկիր (Справочник "Номенклатура" (Общие))' : 'manufacture_country' ,
                            'Вид номенклатуры' : 'product_type' ,
                            'Марка (бренд)' : 'Brand' ,
                            'Номенклатура' : 'Series' ,
                            'Количество' : 'Quantity' ,
                            'Выручка +ԱԱՀ' : 'Revenue+VAT'}


# ========================================================================
#                               SHOP NAMES 
# ========================================================================

shop_names_translation = {
        'Նոր Տուն Էջմիածին (Չգործող)' : 'Etchmiadzin_Nonoperating',
        'Նոր Տուն Մարտունի/Երևանյան(Չգործող)' : 'Martuni/Yerevanyan_Nonoperating',
       '01 Նոր Տուն Արշակունյաց խանութ սրահ' : '01_Arshakunyats_shop',
       '02 Նոր Տուն Դավիթ Անհաղթ խանութ սրահ' : '02_Davit_Anhaght_shop',
       '03 Նոր Տուն Սեբաստիա խանութ սրահ ' : '03_Sebastia_shop',
       '04 Նոր Տուն Վարդանանց խանութ սրահ' : '04_Vardanants_shop',
       '05 Նոր Տուն Կասյան խանութ սրահ' : '05_Kasyan_shop',
       '06 Նոր Տուն Իջևան/Երևանյան խանութ սրահ' : '06_Ijevan/Yerevanyan_shop',
       '07 Նոր Տուն Գյումրի/Շիրակացի խանութ սրահ' : '07_Gyumri/Shirakatsi_shop',
       '08 Նոր Տուն Ավան/Իսահակյան խանութ սրահ' : '08_Avan/Isahakyan_shop',
       '09 Նոր Տուն Դավիթաշեն խանութ սրահ' : '09_Davitashen',
       '10 Նոր Տուն Արտաշատ/Օգոստոսի 23փ․ խանութ սրահ' : '10_Artashat/August23_shop',
       '11 Նոր Տուն Էջմիածին/Վ Առաջինի Խանութ Սրահ' : '11_Etchmiadzin/Vazgen1_shop',
       '12 Նոր Տուն Վանաձոր/Մոսկովյան խանութ սրահ' : '12_Vanadzor/Moskovyan_shop',
       '13 Նոր Տուն Անդրանիկ խանութ սրահ' : '13_Andranik_shop',
       '14 Նոր Տուն Տիգրան Պետրոսյան խանութ սրահ' : '14_Tigran_Petrosyan_shop',
       '15 Նոր Տուն Բաշինջաղյան խանութ սրահ' : '15_Bashinjaghyan_shop',
       '16 Նոր Տուն Նար Դոս խանութ սրահ' : '16_Nar_Dos',
       '17 Նոր Տուն Արմավիր/Երևանյան խանութ սրահ' : '17_Armavir/Yerevanyan_shop',
       '18 Նոր Տուն Դավիթ Բեկ խանութ սրահ' : '18_Davit_Bek_shop',
       '19 Նոր Տուն Ռոստովյան խանութ սրահ' : '19_Rostovyan_shop',
       '19/1 Նոր Տուն Ռոստովյան խանութ սրահ':'19_Rostovyan_shop',
       '20 Նոր Տուն Աշտարակ/Սիսակյան խանութ սրահ' : '20_Ashtarak/Sisakyan_shop',
       '21 Նոր Տուն Աբովյան/Հատիսի խանութ սրահ' : '21_Abovyan/Hatisi_shop',
       '22 Նոր Տուն Կապան խանութ սրահ' : '22_Kapan_shop', 
       '99 Նոր Տուն Օնլայն' : 'Online_shop'}


# ========================================================================
#                               SHOP CITIES 
# ========================================================================

shop_location_mapping = {
         'Etchmiadzin_Nonoperating' : 'Etchmiadzin',
         'Martuni/Yerevanyan_Nonoperating' : 'Martuni',
        '01_Arshakunyats_shop' : 'Yerevan',
        '02_Davit_Anhaght_shop' : 'Yerevan',
        '03_Sebastia_shop' : 'Yerevan',
        '04_Vardanants_shop' : 'Yerevan',
       '05_Kasyan_shop' : 'Yerevan',
        '06_Ijevan/Yerevanyan_shop' : 'Ijevan',
        '07_Gyumri/Shirakatsi_shop' : 'Gyumri',
        '08_Avan/Isahakyan_shop' : 'Yerevan',
        '09_Davitashen' : 'Yerevan',
        '10_Artashat/August23_shop' : 'Artashat',
        '11_Etchmiadzin/Vazgen1_shop' : 'Etchmiadzin',
        '12_Vanadzor/Moskovyan_shop' : 'Vanadzor',
        '13_Andranik_shop' : 'Yerevan',
        '14_Tigran_Petrosyan_shop' : 'Yerevan',
        '15_Bashinjaghyan_shop' : 'Yerevan',
        '16_Nar_Dos' : 'Yerevan',
        '17_Armavir/Yerevanyan_shop' : 'Armavir',
        '18_Davit_Bek_shop' : 'Yerevan',
        '19_Rostovyan_shop' : 'Yerevan',
       '19_Rostovyan_shop' : 'Yerevan',
        '20_Ashtarak/Sisakyan_shop' : 'Ashtarak',
       '21_Abovyan/Hatisi_shop' : 'Abovyan',
       '22_Kapan_shop' : 'Kapan' ,
        'Online_shop' : 'Online'}

# ========================================================================
#                             SHOP -- BRANCH 
# ========================================================================

shop_branch_mapping = {
         'Etchmiadzin_Nonoperating' : 'Etchmiadzin N/O',
         'Martuni/Yerevanyan_Nonoperating' : 'Martuni C.',
        '01_Arshakunyats_shop' : 'Arshakunyats S.',
        '02_Davit_Anhaght_shop' : 'Davit Anhaght S.',
        '03_Sebastia_shop' : 'Sebastia S.',
        '04_Vardanants_shop' : 'Vardanants S.',
       '05_Kasyan_shop' : 'Kasyan S.',
        '06_Ijevan/Yerevanyan_shop' : 'Ijevan C.',
        '07_Gyumri/Shirakatsi_shop' : 'Gyumri C.',
        '08_Avan/Isahakyan_shop' : 'Isahakyan S.',
        '09_Davitashen' : 'Davitashen S.',
        '10_Artashat/August23_shop' : 'Artashat C.',
        '11_Etchmiadzin/Vazgen1_shop' : 'Etchmiadzin C.',
        '12_Vanadzor/Moskovyan_shop' : 'Vanadzor C.',
        '13_Andranik_shop' : 'Andranik S.',
        '14_Tigran_Petrosyan_shop' : 'Tigran Petrosyan S.',
        '15_Bashinjaghyan_shop' : 'Bashinjaghyan S.',
        '16_Nar_Dos' : 'Nar Dos S.',
        '17_Armavir/Yerevanyan_shop' : 'Armavir C.',
        '18_Davit_Bek_shop' : 'Davit Bek S.',
        '19_Rostovyan_shop' : 'Rostovyan S.',
        '20_Ashtarak/Sisakyan_shop' : 'Ashtarak C.',
       '21_Abovyan/Hatisi_shop' : 'Abovyan C.',
       '22_Kapan_shop' : 'Kapan C.' ,
        'Online_shop' : 'Online'}


# ========================================================================
#                               MONTH NAMES
# ========================================================================

month_names_translation = {
                            'Март' : 'March', 
                            'Сентябрь' : 'September', 
                            'Октябрь' : 'October', 
                            'Февраль' : 'February', 
                            'Апрель' : 'April', 
                            'Май' : 'May',
                            'Ноябрь' : 'November', 
                            'Декабрь' : 'December', 
                            'Август' : 'August', 
                            'Июнь' : 'June', 
                            'Июль' : 'July', 
                            'Январь' : 'January'
}

# ========================================================================
#                           MANUFACTURE COUNTRY
# ========================================================================

country_names_translation = {'Ռումինիա' : 'Romania',
                            'Շվեյցարիա' : 'Switzerland',
                            'Իրան' : 'Iran',
                            'Լեհաստան' : 'Poland',
                            'Սերբիա' : 'Serbia',
                            'Իտալիա' : 'Italy',
                            'Պարսկաստան ' : 'Iran', 
                            'Չինաստան' : 'China', 
                            'Ռուսաստան' : 'Russia', 
                            'Իսպանիա' : 'Spain', 
                            'Ուկրաինա' : 'Ukraine',
                            'Բելառուս' : 'Belarus',  
                            'Թաիլանդ' : 'Thailand',  
                            'Իտալիա' : 'Italy', 
                            'Թուրքիա' : 'Turkey', 
                            'Գերմանիա' : 'Germany',
                            'Լատվիա' : 'Latvia', 
                            'Անգլիա' : 'England', 
                            'Ֆրանսիա' : 'France', 
                            'Մալազիա' : 'Malaysia', 
                            'Հնդկաստան' : 'India', 
                            'Պորտուգալիա' : 'Portugal', 
                            'Հունաստան' : 'Greece', 
                            'Հունգարիա' : 'Hungary', 
                            'Բելգիա' : 'Belgium'}



# ========================================================================
#                               PRODUCT TYPE
# ========================================================================

product_type_name_translation = {'Սալիկ հատակի': 'Tile floor',
                                'Օճառաման': 'Soap dish',
                                'Ցնցուղ և ցնցուղի կոմպլեկտներ': 'Shower and shower sets',
                                'Ճկախողովակ գազի': 'Gas hose',
                                'Ճկախողովակ ցնցուղի': 'Shower hose',
                                'Փական մարտկոցի PPR': 'Valve battery PPR',
                                'Ճկախողովակ մետաղական ': 'Pipe metal',
                                'Մետաղական փականներ Arco': 'Metal valves Arco',
                                'Բաշխիչներ': 'Distributors',
                                'Խառնիչ ծորակներ և մասեր': 'Mixer taps and parts',
                                'Մետաղական փականներ': 'Metal valves',
                                'Մետաղյա կցամասեր': 'Metal fittings',
                                'Հոսակ': 'Flow',
                                'Գազի ջրատաքացուցիչ': 'Gas water heater',
                                'Պաստառ լվացվող': 'Wallpaper washable',
                                'Սալիկ պատի': 'Tile wall',
                                'S-կաթսա': 'S-pot',
                                'Նստակոնքի կափարիչ': 'Seat cover',
                                'Պոլիպրոպիլենային կցամասեր PPR': 'Polypropylene fittings PPR',
                                'Լվացարան պահարանով': 'Washbasin with cupboard',
                                'Հայելի': 'A mirror',
                                'Պոլիէթիլենային կցամասեր': 'Polyethylene fittings',
                                'Պրոֆիլ  անկյունակներ': 'Profile corners',
                                'Լվացարան գրանիտե': 'Granite sink',
                                'Գործիքներ խողովակի և կցամասերի': 'Tools for pipe and fittings',
                                'Պահեստամասեր': 'spare parts',
                                'Ֆիլտր': 'Filter:',
                                'Ֆում': 'Fum',
                                'Սիֆոն': 'Siphon',
                                'Գազի կաթսայի ծխատար': 'Gas boiler flue',
                                'Լոգարանի ձող': 'Bathroom rod',
                                'Լրակազմ': 'Suite',
                                'Լվացարան կերամիկական': 'Sink ceramic',
                                'Կախիչ սրբիչի': 'Towel hanger',
                                'Վարագույր/ձող': 'Curtain/Rod',
                                'Միզաման': 'Urinal',
                                'Ամրակներ': 'Fasteners',
                                'F-կաթսա': 'F-boiler',
                                'Քսահարթիչ': 'Lubricant',
                                'պասիվ կոդեր': 'passive codes', #######################################################################
                                'Սոսինձ սալիկի': 'Adhesive tile',
                                'Լվացարան մետաղական': 'Metal sink',
                                'Լվացարանի աքսեսուարներ': 'Wash basin accessories',
                                'Պոլիէթիլենային խողովակներ': 'Polyethylene pipes',
                                'Պոլիպրոպ․ խող․ միջանկյալ ալյումինե շերտով': 'Polypropylene pipes  with an intermediate aluminum layer',
                                'Պոմպ': 'pump',
                                'Պոմպի ավտոմատներ և պահեստամասեր': 'Pump machines and spare parts',
                                'Պոլիպրոպիլենային փականներ PPR': 'Polypropylene valves PPR',
                                'Ֆիլտր PPR': 'Filter PPR',
                                'Լոգարանի օդափոխիչ': 'Bathroom fan',
                                'Գիպսային առաստաղներ': 'Plaster ceilings',
                                'Պանելային մարտկոցներ': 'Panel batteries',
                                'անջատիչ/վարդակ': 'switch/socket',
                                'Էլ․սարքավորումներ': 'Electronic equipment',
                                'Ալյումինե առաստաղներ': 'Aluminum ceilings',
                                'Առաստաղի անկյունակներ և այլ դետալներ': 'Ceiling angles and other details',
                                'Լոգարանի պարագաների հավաքածու': 'A set of bathroom accessories',
                                'Նստակոնք ստանդարտ': 'Seat standard',
                                'Սենյակային Ներկեր': 'Room Paints',
                                'Պաստառ ներկվող': 'Paintable wallpaper',
                                'Չորանոց': 'Dryer',
                                'Շրիշակ պլաստմասե': 'Plastic baseboard',
                                'Պլաստմասե շրիշակի դետալներ': 'Plastic baseboard details',
                                'Լամինատե մանրահատակ': 'Laminate flooring',
                                'Ալյումինե մարտկոցներ': 'Aluminum batteries',
                                'Էլ․ գործիքներ': 'Electronic tools',
                                'Կոյուղու կցամասեր': 'Sewer fittings',
                                'Ընդարձակման բաքեր': 'Expansion tanks',
                                'Պոլիպրոպիլենային խողովակներ': 'Polypropylene pipes',
                                'Մեկուսիչ նյութեր': 'Insulating materials',
                                'Շինարարական քիմիա': 'Construction chemistry',
                                'Գիպսոկարտոնե սալիկների պրոֆիլներ': 'Gypsum board profiles',
                                'Գիպսոկարտոնե սալիկներ': 'Plasterboard tiles',
                                'Էլ․հաղորդալար': 'Electronic conductor',
                                'Ջահեր': 'Torches',
                                'Դռան աքսեսուարներ': 'Door accessories',
                                'Սոսինձ պաստառի': 'wallpaper glue',
                                'լուսատուներ': 'light fixtures',
                                'Էլ․ լամպեր': 'Electronic lamps',
                                'Տակով լոգախցիկ': 'shower cabin with pad', ##########################################################################
                                'Անտակ լոգախցիկ': 'Walk-in shower cabin',
                                'Լամինատի ներքնաշերտ': 'Laminate underlay',
                                'Ցանց օդանցքի,օդատար խողովակներ և լյուկեր': 'Network ventilation, air ducts and hatches',
                                'Չոր շաղախներ/ծեփամածիկներ': 'Dry mortars/plasters',
                                'Մոնտաժ': 'Montage', #####################################################################
                                'Սոսինձներ,հերմետիկներ և այլ նյութեր': 'Adhesives, sealants and other materials',
                                'Լոգնոց': 'A bathroom', ############################################################
                                'Մետաղական ցանցեր': 'Metal networks',
                                'Բիդե': 'Bidet',
                                'Լամինատի մաքրող միջոց': 'Laminate cleaner',
                                'Ճկախողովակ լվ․մեքենայի': 'Washer hose',
                                'Շեմ': 'Threshold:',
                                'Սալիկ գրանիտ': 'Slab granite',
                                'Պենոպլաստե շրջանակներ և դեկորատիվ քիվեր': 'Styrofoam frames and decorative cornices',
                                'Լոգախցիկի տակդիրներ': 'shower cabin pads', ################################################################################
                                'Մետաղապլաստե դուռ/պատուհան': 'Metal-plastic door/window', ###################################################################
                                'Գոֆրե': 'Corrugated',
                                'Պտուտակներ, դյուբելներ և այլ դետալներ': 'Screws, dowels and other details',
                                'Սալիկ դեկոր': 'Tile decor',
                                'Սկավառակներ,գայլիկոններ և բուրեր': 'Discs, drills and burs',
                                'Գոֆրե+գլխիկ': 'Corrugated + head',
                                'Սիֆոնի մասեր': 'Siphon parts',
                                'Զուգարանակոնքի մեխանիզմ/մասեր': 'Toilet mechanism/parts',
                                'Նստակոնք պատի': "Wall hung toilets",
                                'Խաչուկներ և սեպեր': 'Crosses and wedges',
                                'Փականներ Itap': 'Valves Itap', ####################################################################################
                                'Ֆիլտր Itap': 'Filter Itap',      ####################### ES ITAPNERY PITI MNA? XIA ARANDZNACVAC??????###############
                                'Օդահան Itap': 'Air conditioner Itap',###############################################################################
                                'Բաշխիչներ Itap': 'Distributors Itap',
                                'Մետաղյա կցամասեր Itap': 'Metal fittings Itap',
                                'Դռան շրջանակ/շրջանակալ': 'Door frame',
                                'Դուռ միջսենյակային': 'Interior door',
                                'Կպչուն ժապավեններ և ցանցեր': 'Adhesive tapes and nets',
                                'Մանոմետր Itap': 'Manometer Itap', ########### Eli itap 
                                'Լոգ կահույք': 'Log furniture',   ###################################################################
                                'Ծորակ մեկ տեղ․': 'Faucet in one place.', ###################################################################
                                'Ներկ․ պարագաներ': 'Paint: accessories', #########################?????????????????????###############################ATTENTION######
                                'Նախաներկեր': 'Primers',
                                'Զուգարանի խոզանակ': 'Toilet brush',
                                'Կախիչ զուգարանի թղթի': 'Toilet paper hanger',
                                'Ֆեն': 'Fan',
                                'Տակդիր պատի': 'Wall mounted',  ######################################################
                                'Աղբի դույլ': 'Trash can',
                                'Ատամի խոզանակի բաժակ': 'Toothbrush cup',
                                'Ֆենի տակդիր': 'Hair dryer stand',  ##############################################################
                                'Էլեկտրական ջրատաքացուցիչ': 'Electric water heater',
                                'Աքսեսուարներ': 'Accessories',
                                'Ծախսեր': 'Costs:', ##################?????###################
                                'Կափույր Itap': 'Damper Itap', #itap
                                'Պլաստմասե առաստաղների դետալներ': 'Details of plastic ceilings',
                                'Գործիք': 'A tool',
                                'Բազմոց': 'Couch',
                                'Օդորակիչ': 'Air conditioner',
                                'Սեղան/աթոռ': 'Table chair',
                                'Դռան բռնակներ': 'Door handles',
                                'Դուռ մետաղյա': 'Metal door',
                                'Յուղաներկ,լաքեր և լուծիչներ': 'Oil paint, varnishes and solvents',
                                'Լվացքի չորանոց': 'Washer dryer', ###########################################################
                                'Աստիճան': 'Լadder',
                                'Պլաստմասե առաստաղներ': 'Plastic ceilings',
                                'Տեղական': 'local', ###########################################################################
                                'Տեղադրում': 'Installation:',
                                'Կոյուղու խողովակներ': 'Sewer pipes',
                                'Խողովակի մեկուսիչ': 'Pipe insulator',
                                'Ճկախողովակ ռետինե ': 'Rubber hose',
                                'Գունանյութեր': 'Pigments',
                                'Շրիշակ լամինացված': 'Laminated baseboard',
                                'Բեռնափոխադրում': 'transportation',
                                'Բանվորական հագուստ/ձեռնոցներ': 'Work clothes/gloves',
                                'Պոլիէթիլենային փականներ': 'Polyethylene valves',
                                'Կափույր': 'Damper', ####################################################################################
                                'Թերմոստատ': 'Thermostat',
                                'Քիմիական նյութեր և մաքրող միջոցներ': 'Chemicals and cleaning substances',
                                'Հիդրոմերսող լոգնոց': 'Hydromassage bath', ##################################################################
                                'Սալիկ մոզաիկա': 'Tile mosaic',
                                'Շերտավարագույրներ': 'Blinds',
                                'Էլ․տաքացուցիչ': 'Electronic heater',
                                'Նստակոնքի պահեստամասեր': 'Seat spare parts', ###############################################################
                                'Լվացարանի պահեստամասեր': 'Sink spare parts', #################################################################
                                'էլ․կաբել-կանալ': 'Electronic cable-canal',
                                'Հղկաթուղթ': 'Sandpaper',
                                'Սալիկ գոտի': 'Tile belt',
                                'Փական մարտկոցի': 'Close the battery', #########################################################################
                                'Բոյլեր': 'Boiler',
                                'Լամ․ խցանե ներքնաշերտ': 'cork lining', #####################################################
                                'Աէորոզոլային ներկեր': 'Aerosol paints',
                                'Պատվերով կահույք': 'Custom furniture',
                                'Դեկորատիվ ներկեր': 'Decorative paints',
                                'Էլեկտրոդ և մետաղալար': 'Electrode and wire',
                                'Կոնվեկտորային մարտկոցներ': 'Convector batteries',
                                'Աքսեսուրներ(բռնակներ)': 'Accessories (handles)',
                                'Հոսակի լրացում-դետալ': 'Filling-detail of Hosak', ############################################################################
                                'Դարակներ': 'Shelves',
                                'Ծառայություն': 'Service:',
                                '3D դեկորատիվ պանելներ': '3D decorative panels',
                                'Ֆանկոյլ': 'Fancoil',
                                'Բեռնասայլակ': 'Truck',
                                'Հիդրոմերսող լոգախցիկ': 'Hydromassage shower cabin',
                                'Պոլիէթիլենային խող․ միջանկյալ ալումինե շերտով': 'Polyethylene pipe with an intermediate aluminum layer',
                                'Էլ․ կաթսա': 'Electronic boiler',
                                'Կառավարման վահանակներ': 'Control panels',
                                'Գործիքներ (սպասարկման կենտրոն)': 'Tools (service center)',
                                'Կաթսայի պահեստամասեր': 'Boiler spare parts',
                                'Տեռասսային հատակներ': 'Terrace floors',
                                'Ծորակի պահեստամասեր': 'Faucet spare parts',
                                'Մարտկոցներ և լիցքավորման սարքեր': 'Batteries and charging devices',
                                'Արդուկի տախտակ':'Ironing board',
                                 'R-կաթսա':'R-boiler',
                                'Մահճակալ':'bed',
                                 'Տեքստիլ':'Textile',
                                 'Այլ պահեստամասեր':'Other spare parts',
                                 'Էլ․տաքացուցիչ':'Electronic heater',
                                'համալրող դետալներ':'complementary parts',
                                'Վերանորոգում' : 'Repair',
                                'Դուշ․պանել' : 'Shower Panel',
                                'Լոգարանի գորգ' : 'Bathroom mat',
                                'Ներքնակ' : 'Mattress',
                                'Ստացված Ծառայություն':'',
                                'Լոգախցիկի պահեստամասեր' : 'Shower cabin spare parts',
                                'Էլ․ տաքացուցիչ':'Electronic heater',
                                'Մետաղական ուղղանկյուն խողովակներ' : 'Metal rectangular pipes'
                                }



# ========================================================================
#                                BRAND
# ========================================================================

brand_name_translation = {' Շեն' : 'Shen',
                            'Շեն' : 'Shen',
                            'Комфорт' : 'Komfort',
                            'Славянские обои' : 'Slovyanski Shpalery',
                            'Эра' : 'ERA',
                            'ФАЗА' : 'FAZA',
                            'АРИКОН' : 'ARIKON',
                            'Мелодия сна' : 'Melodiya sna',
                            'Ресанта' : 'Resanta',
                            'Թերմոսթայլ' : 'Thermostyle',
                            ' ԾԻԱԾԱՆ' : 'TSIATSAN',
                            'МОФ' : 'MOF',
                            'Коллекция' : 'Kollektsiya',
                            'ԾԻԱԾԱՆ' : 'TSIATSAN'}



# ========================================================================
#                               DATA LOADING
# ========================================================================