"""Few-shot Q&A examples for the analyst-facing RAG.

Each entry shows the LLM the expected answer style: short, content-bearing,
with inline [n] citations per the SYSTEM_PROMPT in query.py.

Used by query.py to prepend examples before the real retrieved context.
"""
from __future__ import annotations


# 50 hand-written examples. All authors / handles / events are fictional.
# Distribution:
#   Languages: ~20 English, ~15 Hebrew, ~15 Arabic
#   Question types (5 each):
#     1.  Attribution           ("who said what")
#     2.  Sentiment / mood
#     3.  Trend / volume
#     4.  Threat / incitement detection
#     5.  Network / actor amplification
#     6.  Event timeline / chronology
#     7.  Geographic
#     8.  Disinformation / claim verification
#     9.  Comparative (cross-language / cross-platform)
#     10. Posts-don't-answer (rule #4 of SYSTEM_PROMPT)
FEW_SHOT_QA_EXAMPLES = [
    # ============================================================
    # 1. ATTRIBUTION
    # ============================================================
    {
        "question": "Who is calling for protests in Haifa next weekend?",
        "posts": [
            {"n": 1, "source": "facebook_posts", "author": "Haifa Workers Forum",
             "date": "2025-09-12",
             "content": "Mass rally Saturday 18:00 at Paris Square. We will not stay silent about the port layoffs."},
            {"n": 2, "source": "twitter_posts", "author": "@haifa_left",
             "date": "2025-09-13",
             "content": "Join us this Saturday in Haifa. Buses leaving from Hadar at 17:00. Spread the word."},
            {"n": 3, "source": "telegram_posts", "author": "Northern Voice",
             "date": "2025-09-13",
             "content": "Reposting the Haifa Workers Forum call for Saturday. We stand with the dockworkers."},
            {"n": 4, "source": "instagram_posts", "author": "tel_aviv_foodie",
             "date": "2025-09-12",
             "content": "Best hummus in Wadi Nisnas, hands down. Going back tomorrow."},
        ],
        "answer": (
            "The Haifa Workers Forum issued the original call for a Saturday 18:00 rally at Paris "
            "Square over the port layoffs [1]. The Twitter account @haifa_left is mobilizing buses "
            "from Hadar at 17:00 [2], and the Telegram channel Northern Voice is amplifying the "
            "Workers Forum call in solidarity with the dockworkers [3]. Post [4] is unrelated "
            "(restaurant content)."
        ),
    },
    {
        "question": "Which accounts announced the teachers' union strike?",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@TeachersUnionIL",
             "date": "2026-01-08",
             "content": "Strike confirmed for Sunday. All elementary and middle schools nationwide. Demands: pay raise and class-size cap."},
            {"n": 2, "source": "facebook_posts", "author": "Parents4Teachers",
             "date": "2026-01-08",
             "content": "We support Sunday's strike. Parents will picket together with teachers at 8 AM at the Education Ministry."},
            {"n": 3, "source": "telegram_posts", "author": "EduWatch",
             "date": "2026-01-09",
             "content": "Confirming: Teachers Union announced full strike Sunday. High schools join Monday if no deal by midnight."},
        ],
        "answer": (
            "@TeachersUnionIL formally announced a nationwide strike of elementary and middle "
            "schools for Sunday, citing pay raises and class-size caps as the demands [1]. The "
            "Parents4Teachers Facebook group endorsed the strike and called a parallel 8 AM picket "
            "at the Education Ministry [2]. EduWatch on Telegram corroborated the announcement and "
            "added that high schools would join Monday absent a midnight deal [3]."
        ),
    },
    {
        "question": "מי קרא להפגנות בכיכר רבין הערב?",
        "posts": [
            {"n": 1, "source": "facebook_posts", "author": "המטה למאבק",
             "date": "2025-11-04",
             "content": "כולם לכיכר רבין הערב 19:00. לא נשתוק מול חוק ההסדרים. הביאו דגלים."},
            {"n": 2, "source": "twitter_posts", "author": "@noa_activist",
             "date": "2025-11-04",
             "content": "מתראים בכיכר רבין הערב בשבע. שתפו, חשוב שיגיעו רבים."},
            {"n": 3, "source": "telegram_posts", "author": "ערוץ ההתנגדות",
             "date": "2025-11-04",
             "content": "המטה למאבק מארגן הערב הפגנה בכיכר רבין. נציגי הפורומים השכונתיים יתאספו בצד דרום."},
            {"n": 4, "source": "instagram_posts", "author": "tlv.streetphoto",
             "date": "2025-11-03",
             "content": "אור ירח מעל פלורנטין אתמול בלילה. שכונה שלא ישנה."},
        ],
        "answer": (
            "‏'המטה למאבק' פרסם את הקריאה המרכזית להפגנה הערב בכיכר רבין בשעה 19:00 בעקבות "
            "חוק ההסדרים [1]. החשבונית @noa_activist בטוויטר קוראת להגיע לאותה הפגנה בשבע [2], "
            "ו'ערוץ ההתנגדות' בטלגרם מאשר את האירוע ומציין כי נציגי הפורומים השכונתיים יתאספו "
            "בצד דרום [3]. פוסט [4] אינו רלוונטי – צילום רחוב מפלורנטין."
        ),
    },
    {
        "question": "מי מפיץ את הקריאה לסגירת בתי הספר מחר?",
        "posts": [
            {"n": 1, "source": "telegram_posts", "author": "הורים מתאחדים",
             "date": "2026-02-19",
             "content": "מחר אנחנו לא שולחים ילדים לבית הספר. שביתת מחאה על מצב המקלטים. שתפו לכל הקבוצות."},
            {"n": 2, "source": "facebook_posts", "author": "ועד הורים מרכז",
             "date": "2026-02-19",
             "content": "אנו תומכים בקריאה של 'הורים מתאחדים'. בקשה לסגן ראש העיר נשלחה הבוקר."},
            {"n": 3, "source": "twitter_posts", "author": "@yossi_dad",
             "date": "2026-02-19",
             "content": "מחר לא שולחים את הילדים. די לאוזלת היד. כל הורה אחראי – בבקשה הצטרפו."},
        ],
        "answer": (
            "קבוצת 'הורים מתאחדים' בטלגרם יזמה את הקריאה שלא לשלוח ילדים לבית הספר מחר במחאה "
            "על מצב המקלטים [1]. ועד הורים מרכז בפייסבוק מאמץ את הקריאה ומדווח שנשלחה בקשה "
            "פורמלית לסגן ראש העיר [2]. החשבון הפרטי @yossi_dad מצטרף ומפיץ את אותה קריאה [3]."
        ),
    },
    {
        "question": "من يدعو إلى التظاهر في غزة الأسبوع المقبل؟",
        "posts": [
            {"n": 1, "source": "facebook_posts", "author": "صوت غزة",
             "date": "2025-07-21",
             "content": "نطالب بمسيرة شعبية يوم الجمعة بعد الصلاة من ميدان فلسطين. لا للصمت أمام انقطاع الكهرباء."},
            {"n": 2, "source": "twitter_posts", "author": "@gaza_youth_now",
             "date": "2025-07-22",
             "content": "نشارك دعوة صوت غزة لمسيرة الجمعة. لنخرج جميعاً ونرفع صوتنا."},
            {"n": 3, "source": "telegram_posts", "author": "تجمع شباب القطاع",
             "date": "2025-07-22",
             "content": "ندعم التحرك يوم الجمعة. سنوفر حافلات من شمال القطاع نحو ميدان فلسطين."},
            {"n": 4, "source": "instagram_posts", "author": "ramallah_eats",
             "date": "2025-07-21",
             "content": "أفضل كنافة نابلسية في رام الله، جربوها مساءً."},
        ],
        "answer": (
            "صفحة 'صوت غزة' على فيسبوك أطلقت الدعوة الأصلية لمسيرة شعبية يوم الجمعة بعد الصلاة "
            "من ميدان فلسطين احتجاجاً على انقطاع الكهرباء [1]. حساب @gaza_youth_now على تويتر "
            "يعيد نشر الدعوة ويحث على المشاركة [2]، فيما أعلن 'تجمع شباب القطاع' على تيليغرام "
            "عن تنظيم حافلات من شمال القطاع لدعم التحرك [3]. المنشور [4] غير ذي صلة (محتوى "
            "طعام من رام الله)."
        ),
    },

    # ============================================================
    # 2. SENTIMENT / MOOD
    # ============================================================
    {
        "question": "How is the public reacting to the new fuel tax?",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@drivers_il",
             "date": "2025-10-02",
             "content": "Another tax on the working class. I already pay 7.50 a liter, where does it end?"},
            {"n": 2, "source": "facebook_posts", "author": "Maya R.",
             "date": "2025-10-02",
             "content": "This fuel hike will kill small delivery businesses. Three of my drivers said they're quitting."},
            {"n": 3, "source": "twitter_posts", "author": "@green_future_il",
             "date": "2025-10-03",
             "content": "Finally serious pricing on fossil fuels. Public transit funding has to be next."},
            {"n": 4, "source": "telegram_posts", "author": "Markets Daily",
             "date": "2025-10-03",
             "content": "Fuel-tax bill passed at 02:30. Effective Nov 1. Diesel +0.35, gasoline +0.42 per liter."},
        ],
        "answer": (
            "Reaction is mostly negative: @drivers_il complains that ordinary drivers already pay "
            "around 7.50 per liter and questions where the burden ends [1], and Maya R. warns the "
            "hike will push small delivery businesses to fold, with three of her drivers already "
            "quitting [2]. A minority view from @green_future_il welcomes the move as overdue "
            "fossil-fuel pricing and calls for transit investment next [3]. Markets Daily provides "
            "neutral confirmation that the bill passed at 02:30 with diesel up 0.35 and gasoline "
            "up 0.42 per liter from Nov 1 [4]."
        ),
    },
    {
        "question": "What's the mood around the cabinet reshuffle?",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@politics_watch",
             "date": "2026-03-11",
             "content": "Three ministers replaced overnight. Coalition partners furious — they weren't consulted."},
            {"n": 2, "source": "facebook_posts", "author": "Daniel B.",
             "date": "2026-03-11",
             "content": "Finally fresh faces in finance and transport. Maybe now something will move."},
            {"n": 3, "source": "telegram_posts", "author": "Knesset Insider",
             "date": "2026-03-12",
             "content": "Senior MK described mood inside the coalition as 'toxic'. Vote of confidence likely next week."},
        ],
        "answer": (
            "The dominant mood is anger and instability: @politics_watch reports that coalition "
            "partners are furious because they were not consulted on the overnight replacement of "
            "three ministers [1], and Knesset Insider quotes a senior MK calling the internal "
            "atmosphere 'toxic' with a confidence vote expected next week [3]. A more positive "
            "civilian voice, Daniel B., welcomes the new finance and transport faces in hopes of "
            "movement on stalled files [2]."
        ),
    },
    {
        "question": "איזו תחושה משדרים תושבי שדרות לאחר הירי הלילה?",
        "posts": [
            {"n": 1, "source": "facebook_posts", "author": "תושבי שדרות מדברים",
             "date": "2025-12-04",
             "content": "לילה נוסף ללא שינה. הילדים ישנים בממ\"ד כבר שבוע. אנחנו עייפים, לא מפחדים – עייפים."},
            {"n": 2, "source": "twitter_posts", "author": "@sderot_news",
             "date": "2025-12-04",
             "content": "תחושה כללית בעיר: תסכול עמוק מהממשלה, לא מהצבא. דורשים פתרון של קבע."},
            {"n": 3, "source": "telegram_posts", "author": "ערוץ הדרום",
             "date": "2025-12-04",
             "content": "סיור באחד הרחובות הראשיים: חנויות סגורות, מעט אנשים בחוץ. רוח שקטה ומורטת עצבים."},
            {"n": 4, "source": "instagram_posts", "author": "rina.shoots",
             "date": "2025-12-04",
             "content": "ילדה מציירת לב על דלת הממ\"ד. שדרות שלי, חזקה ועייפה."},
        ],
        "answer": (
            "התחושה השלטת היא עייפות ותסכול ולא בהלה. דף 'תושבי שדרות מדברים' מתאר לילה נוסף "
            "ללא שינה וילדים שישנים שבוע במרחב המוגן – 'לא מפחדים, עייפים' [1]. @sderot_news "
            "מתארגנים סביב כעס על הממשלה (לא על הצבא) ודרישה לפתרון של קבע [2]. 'ערוץ הדרום' "
            "מצלם רחובות ריקים וחנויות סגורות, באווירה שקטה ומורטת עצבים [3]. הצלמת rina.shoots "
            "ממסגרת זאת בתמונה רגשית של ילדה המציירת לב על דלת הממ\"ד [4]."
        ),
    },
    {
        "question": "كيف يتفاعل الناس مع قرار رفع الدعم عن الخبز؟",
        "posts": [
            {"n": 1, "source": "facebook_posts", "author": "أم كريم",
             "date": "2025-06-15",
             "content": "كيف سنطعم أولادنا؟ ربطة الخبز ارتفعت ضعفين منذ الصباح. هذا ظلم."},
            {"n": 2, "source": "twitter_posts", "author": "@cairo_streets",
             "date": "2025-06-15",
             "content": "طوابير طويلة أمام المخابز اليوم. الناس غاضبون والحكومة صامتة."},
            {"n": 3, "source": "telegram_posts", "author": "اقتصاديون مستقلون",
             "date": "2025-06-16",
             "content": "رفع الدعم خطوة قاسية لكنها ضرورية لخفض العجز. البديل كان أسوأ."},
            {"n": 4, "source": "instagram_posts", "author": "yara_cooks",
             "date": "2025-06-15",
             "content": "وصفة خبز منزلي اقتصادي. شاركوها مع أهلكم."},
        ],
        "answer": (
            "ردود الفعل غاضبة في الغالب: 'أم كريم' تتساءل كيف ستطعم أبناءها بعد أن تضاعف سعر "
            "ربطة الخبز منذ الصباح وتصف القرار بالظلم [1]، و@cairo_streets يرصد طوابير طويلة "
            "أمام المخابز وغضباً شعبياً مقابل صمت حكومي [2]. على الجانب الآخر، قناة 'اقتصاديون "
            "مستقلون' ترى أن الخطوة قاسية لكنها ضرورية لخفض العجز [3]. المنشور [4] محايد "
            "ويعرض وصفة خبز منزلي اقتصادي."
        ),
    },
    {
        "question": "ما رأي الناس في خطاب رئيس الوزراء أمس؟",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@analyst_sam",
             "date": "2026-04-02",
             "content": "خطاب فارغ من المضمون، أرقام بدون خطة. أين الإصلاح الذي وعد به؟"},
            {"n": 2, "source": "facebook_posts", "author": "محمود ع.",
             "date": "2026-04-02",
             "content": "أخيراً سمعنا اعترافاً بحجم الأزمة. لا أوافق على كل شيء لكن الصراحة مهمة."},
            {"n": 3, "source": "telegram_posts", "author": "غرفة الأخبار",
             "date": "2026-04-02",
             "content": "الخطاب استمر 47 دقيقة، أبرز نقطة: تعليق مؤقت لضريبة الشركات الصغيرة لمدة عام."},
        ],
        "answer": (
            "الآراء منقسمة: @analyst_sam يصف الخطاب بأنه فارغ من المضمون وأرقام بلا خطة، "
            "ويتساءل عن وعود الإصلاح [1]، بينما يرى محمود ع. أن مجرد الاعتراف بحجم الأزمة "
            "كان خطوة إيجابية حتى لو لم يوافق على كل التفاصيل [2]. 'غرفة الأخبار' تقدم "
            "تلخيصاً محايداً يفيد بأن الخطاب استمر 47 دقيقة وأبرز إعلان فيه كان تعليق ضريبة "
            "الشركات الصغيرة لمدة عام [3]."
        ),
    },

    # ============================================================
    # 3. TREND / VOLUME
    # ============================================================
    {
        "question": "What topics spiked on Telegram this week?",
        "posts": [
            {"n": 1, "source": "telegram_posts", "author": "Trends Watch",
             "date": "2026-02-08",
             "content": "Top movers this week: #FuelTax (+340%), #HostageDeal (+210%), #CoachFired (+180%), #StormGabriel (+95%)."},
            {"n": 2, "source": "telegram_posts", "author": "Politico Israel",
             "date": "2026-02-07",
             "content": "Fuel-tax channels are exploding. Three new mobilization groups hit 10k members in 48 hours."},
            {"n": 3, "source": "telegram_posts", "author": "Sports Updates",
             "date": "2026-02-06",
             "content": "Coach Mizrahi fired after Sunday's loss. Channel grew by 4,200 members overnight."},
            {"n": 4, "source": "telegram_posts", "author": "Weather Now",
             "date": "2026-02-08",
             "content": "Storm Gabriel update: heavy rain through Sunday in the north, snow above 800m."},
        ],
        "answer": (
            "Trends Watch ranks the top movers as #FuelTax (+340%), #HostageDeal (+210%), "
            "#CoachFired (+180%), and #StormGabriel (+95%) [1]. Politico Israel notes that fuel-tax "
            "channels are particularly explosive, with three new mobilization groups reaching 10k "
            "members in 48 hours [2]. Sports Updates attributes the coach-firing spike to Coach "
            "Mizrahi being dismissed after Sunday's loss [3], and Weather Now confirms ongoing "
            "Storm Gabriel coverage with heavy rain and snow above 800m [4]."
        ),
    },
    {
        "question": "Which hashtags gained traction yesterday?",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@hashtag_radar",
             "date": "2025-11-21",
             "content": "Trending in IL yesterday: #BringThemHome (1.2M mentions), #PriceOfBread (380K), #DerbyDrama (210K)."},
            {"n": 2, "source": "twitter_posts", "author": "@civic_data",
             "date": "2025-11-21",
             "content": "#BringThemHome overtook #FuelTax as the top political tag for the first time in 6 weeks."},
            {"n": 3, "source": "facebook_posts", "author": "MediaLab IL",
             "date": "2025-11-21",
             "content": "Bread-price tag is largely organic — driven by neighborhood groups, not influencers."},
        ],
        "answer": (
            "@hashtag_radar reports that #BringThemHome (1.2M mentions), #PriceOfBread (380K), and "
            "#DerbyDrama (210K) were the top trending hashtags yesterday in Israel [1]. "
            "@civic_data adds that #BringThemHome overtook #FuelTax as the top political tag for "
            "the first time in six weeks [2]. MediaLab IL characterizes the #PriceOfBread surge as "
            "organic and driven by neighborhood groups rather than influencers [3]."
        ),
    },
    {
        "question": "מה היו הנושאים הבולטים בשיח השבוע?",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@trend_il",
             "date": "2025-08-16",
             "content": "השבוע: עסקת החטופים (1.4 מיליון אזכורים), מחיר הדלק (310K), גמר הגביע (180K)."},
            {"n": 2, "source": "facebook_posts", "author": "מבט תקשורתי",
             "date": "2025-08-15",
             "content": "השיח על עסקת החטופים מוביל בפער גדול. עלייה של 220% מהשבוע הקודם."},
            {"n": 3, "source": "telegram_posts", "author": "ניטור רשתות",
             "date": "2025-08-16",
             "content": "מחיר הדלק תופס תאוצה בקבוצות שכונתיות, לא רק בחשבונות פוליטיים."},
            {"n": 4, "source": "instagram_posts", "author": "tlv_streets",
             "date": "2025-08-16",
             "content": "אווירת קיץ ברוטשילד הערב, אנשים, מוזיקה, חיים."},
        ],
        "answer": (
            "‏@trend_il מדווח שהנושאים הבולטים השבוע היו עסקת החטופים (1.4 מיליון אזכורים), "
            "מחיר הדלק (310K) וגמר הגביע (180K) [1]. 'מבט תקשורתי' מציין כי השיח על עסקת "
            "החטופים מוביל בפער גדול, עם עלייה של 220% מהשבוע הקודם [2]. 'ניטור רשתות' מוסיף "
            "שמחיר הדלק תופס תאוצה דווקא בקבוצות שכונתיות ולא רק בחשבונות פוליטיים [3]. פוסט "
            "[4] אינו רלוונטי – צילום אווירה ברוטשילד."
        ),
    },
    {
        "question": "אילו האשטגים תפסו תאוצה בטוויטר אתמול?",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@hashradar_il",
             "date": "2026-05-03",
             "content": "האשטגים מובילים אתמול: #החזירו_אותם_עכשיו, #די_להפיכה, #אסון_בכביש6."},
            {"n": 2, "source": "twitter_posts", "author": "@netivot_data",
             "date": "2026-05-03",
             "content": "‏#אסון_בכביש6 הגיע ל-90 אלף אזכורים תוך 6 שעות מהדיווח הראשון."},
            {"n": 3, "source": "facebook_posts", "author": "כתבת רשת",
             "date": "2026-05-03",
             "content": "‏#די_להפיכה הוא ההאשטג הפוליטי הוותיק ביותר עם פעילות אתמול – חזר לטרנד אחרי שבועיים שקטים."},
        ],
        "answer": (
            "‏@hashradar_il מדווח כי שלושת ההאשטגים שהובילו אתמול היו #החזירו_אותם_עכשיו, "
            "#די_להפיכה ו#אסון_בכביש6 [1]. @netivot_data מצביע על #אסון_בכביש6 כצומח במיוחד – "
            "כ-90 אלף אזכורים תוך שש שעות מהדיווח הראשוני [2]. 'כתבת רשת' מציינת ש#די_להפיכה "
            "חזר לטרנד אתמול לאחר כשבועיים של דעיכה [3]."
        ),
    },
    {
        "question": "ما الموضوعات التي ارتفع الحديث عنها هذا الأسبوع؟",
        "posts": [
            {"n": 1, "source": "telegram_posts", "author": "رصد الشبكات",
             "date": "2025-10-19",
             "content": "أبرز الموضوعات هذا الأسبوع: صفقة الأسرى (+260%), رفع أسعار الوقود (+150%), فضيحة كرة القدم (+90%)."},
            {"n": 2, "source": "twitter_posts", "author": "@arab_trends",
             "date": "2025-10-18",
             "content": "صفقة الأسرى تتصدر الترند العربي للأسبوع الثاني على التوالي."},
            {"n": 3, "source": "facebook_posts", "author": "مرصد الإعلام",
             "date": "2025-10-19",
             "content": "أسعار الوقود تنتشر بقوة في مجموعات الأحياء وليس فقط القنوات السياسية."},
        ],
        "answer": (
            "تظهر بيانات 'رصد الشبكات' أن صفقة الأسرى (+260%)، ورفع أسعار الوقود (+150%)، "
            "وفضيحة كرة القدم (+90%) كانت الموضوعات الأبرز هذا الأسبوع [1]. @arab_trends يؤكد "
            "أن صفقة الأسرى تتصدر الترند العربي للأسبوع الثاني على التوالي [2]، فيما يلاحظ "
            "'مرصد الإعلام' أن نقاش أسعار الوقود ينتشر بقوة داخل مجموعات الأحياء وليس فقط "
            "القنوات السياسية [3]."
        ),
    },

    # ============================================================
    # 4. THREAT / INCITEMENT
    # ============================================================
    {
        "question": "Are there explicit calls for violence against the police chief?",
        "posts": [
            {"n": 1, "source": "telegram_posts", "author": "Anonymous17",
             "date": "2025-09-30",
             "content": "Whoever finds his car gets a reward. He won't make it to the next press conference."},
            {"n": 2, "source": "twitter_posts", "author": "@law_and_order_il",
             "date": "2025-09-30",
             "content": "I disagree with the chief but threats are unacceptable. Report any specific calls to the police."},
            {"n": 3, "source": "facebook_posts", "author": "Civic Watch",
             "date": "2025-09-30",
             "content": "Three Telegram channels published photos of his home address tonight. This is dangerous escalation."},
            {"n": 4, "source": "telegram_posts", "author": "Patriot Voice",
             "date": "2025-09-29",
             "content": "Time to hold him accountable in the streets. Peaceful protest at HQ tomorrow at 10."},
        ],
        "answer": (
            "Yes, post [1] from 'Anonymous17' is an explicit threat: it offers a reward for "
            "locating the chief's car and states 'he won't make it to the next press conference'. "
            "Civic Watch warns that three Telegram channels published photos of his home address, "
            "describing this as a dangerous escalation [3]. By contrast @law_and_order_il "
            "explicitly condemns threats and urges reporting [2], while Patriot Voice's call is a "
            "peaceful protest at HQ at 10 AM rather than a violent threat [4]."
        ),
    },
    {
        "question": "Have any accounts threatened the courthouse or judges?",
        "posts": [
            {"n": 1, "source": "telegram_posts", "author": "JusticeForAll",
             "date": "2025-12-22",
             "content": "If they rule against us tomorrow, we will burn the building. No more empty words."},
            {"n": 2, "source": "twitter_posts", "author": "@rule_of_law",
             "date": "2025-12-22",
             "content": "Concerning posts circulating about tomorrow's ruling. Reported the worst ones to the cyber unit."},
            {"n": 3, "source": "facebook_posts", "author": "Bar Association",
             "date": "2025-12-22",
             "content": "We strongly condemn any threat against judges. The judiciary must be able to work without fear."},
        ],
        "answer": (
            "Yes — 'JusticeForAll' on Telegram explicitly threatens to burn the courthouse if "
            "tomorrow's ruling goes against them [1]. @rule_of_law confirms that concerning threat "
            "content is circulating and says they reported the worst posts to the cyber unit [2]. "
            "The Bar Association issued a statement strongly condemning threats against judges and "
            "defending judicial independence [3]."
        ),
    },
    {
        "question": "האם יש קריאות לפגיעה בשופטים?",
        "posts": [
            {"n": 1, "source": "telegram_posts", "author": "מעגל הפעולה",
             "date": "2026-01-30",
             "content": "מי שייתן יד לפסק הדין הזה ישלם מחיר. אנחנו יודעים איפה גרים. השעון מתקתק."},
            {"n": 2, "source": "twitter_posts", "author": "@israel_legal",
             "date": "2026-01-30",
             "content": "ראינו השבוע מספר פוסטים מאיימים על שופטות. דיווחנו ליאחב\"ל. צריך להגן עליהן."},
            {"n": 3, "source": "facebook_posts", "author": "פורום השופטים",
             "date": "2026-01-30",
             "content": "אנו מגנים בתוקף את הקריאות האלימות נגד עמיתינו. עצמאות השפיטה אינה משא ומתן."},
        ],
        "answer": (
            "כן – פוסט [1] מ'מעגל הפעולה' בטלגרם מהווה איום מפורש: 'מי שייתן יד לפסק הדין הזה "
            "ישלם מחיר. אנחנו יודעים איפה גרים'. @israel_legal מאשר שמספר פוסטים מאיימים נצפו "
            "ושדווחו ליחידת יאחב\"ל [2], ופורום השופטים פרסם הודעת גינוי ועומד על עצמאות השפיטה "
            "[3]."
        ),
    },
    {
        "question": "هل هناك تحريض ضد رجال الشرطة؟",
        "posts": [
            {"n": 1, "source": "telegram_posts", "author": "صوت الشارع",
             "date": "2025-08-09",
             "content": "إذا اعتدوا على شبابنا الليلة، سنرد بقوة. كل واحد منا يعرف من هم."},
            {"n": 2, "source": "twitter_posts", "author": "@khalil_speaks",
             "date": "2025-08-09",
             "content": "نرفض العنف من أي جهة. الاحتجاج حق، التحريض جريمة."},
            {"n": 3, "source": "facebook_posts", "author": "لجنة حقوق",
             "date": "2025-08-09",
             "content": "رصدنا منشورات تحرّض على عناصر الشرطة وتنشر صور وجوههم. هذا خط أحمر."},
            {"n": 4, "source": "instagram_posts", "author": "yafa_pics",
             "date": "2025-08-09",
             "content": "غروب يافا الليلة. اللون البرتقالي يغطي البحر."},
        ],
        "answer": (
            "نعم – المنشور [1] من 'صوت الشارع' يتضمن تهديداً مباشراً برد عنيف ضد عناصر "
            "الشرطة وعبارة 'كل واحد منا يعرف من هم'. 'لجنة حقوق' ترصد منشورات تحرّض على عناصر "
            "الشرطة وتنشر صور وجوههم وتعتبر ذلك خطاً أحمر [3]. في المقابل @khalil_speaks يرفض "
            "العنف صراحةً ويصف التحريض بالجريمة [2]، والمنشور [4] غير ذي صلة (صورة غروب)."
        ),
    },
    {
        "question": "هل توجد تهديدات لقضاة المحكمة العليا؟",
        "posts": [
            {"n": 1, "source": "telegram_posts", "author": "حركة المواطن الحر",
             "date": "2026-02-05",
             "content": "نعرف عناوينهم وعناوين أبنائهم. الحساب لن يطول."},
            {"n": 2, "source": "twitter_posts", "author": "@constitution_now",
             "date": "2026-02-05",
             "content": "أبلغنا الجهات المختصة عن قنوات تنشر صوراً لمنازل القضاة. هذا ليس نقاشاً سياسياً."},
            {"n": 3, "source": "facebook_posts", "author": "نادي القضاء",
             "date": "2026-02-06",
             "content": "ندين بشدة كل تهديد ضد قضاتنا. لا مساومة على استقلال القضاء."},
        ],
        "answer": (
            "نعم — المنشور [1] من 'حركة المواطن الحر' تهديد صريح يقول 'نعرف عناوينهم وعناوين "
            "أبنائهم. الحساب لن يطول'. @constitution_now يؤكد أن قنوات تنشر صوراً لمنازل القضاة "
            "وأنه أبلغ الجهات المختصة [2]، و'نادي القضاء' أصدر بياناً يدين التهديدات ويتمسك "
            "باستقلال القضاء [3]."
        ),
    },

    # ============================================================
    # 5. NETWORK / ACTOR ANALYSIS
    # ============================================================
    {
        "question": "Which accounts are amplifying the 'shadow vote' narrative?",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@truth_voter_il",
             "date": "2026-03-20",
             "content": "Shadow vote is real. 40,000 ballots can't be explained. Demand a recount."},
            {"n": 2, "source": "telegram_posts", "author": "Patriot Network",
             "date": "2026-03-20",
             "content": "Reposting @truth_voter_il. Every channel needs to push #ShadowVote until they answer."},
            {"n": 3, "source": "twitter_posts", "author": "@elections_check",
             "date": "2026-03-21",
             "content": "Same identical 'shadow vote' phrasing in 240 accounts created in February. Coordinated."},
            {"n": 4, "source": "facebook_posts", "author": "Citizen Watch",
             "date": "2026-03-21",
             "content": "We mapped the spread: three Telegram channels seed the content, ~50 X accounts amplify it within minutes."},
        ],
        "answer": (
            "@truth_voter_il is one of the seeding voices, asserting unexplained ballots and "
            "demanding a recount [1], and Patriot Network on Telegram reposts that account and "
            "instructs other channels to push #ShadowVote [2]. @elections_check identifies "
            "coordinated behavior — 240 February-created accounts using identical 'shadow vote' "
            "phrasing [3]. Citizen Watch maps the structure: three Telegram channels seed the "
            "content while ~50 X accounts amplify within minutes [4]."
        ),
    },
    {
        "question": "Who are the main amplifiers of the Iran-strike rumor?",
        "posts": [
            {"n": 1, "source": "telegram_posts", "author": "DefenseLeaks",
             "date": "2025-11-09",
             "content": "Sources confirm a strike package being prepared. Watch the next 72 hours."},
            {"n": 2, "source": "twitter_posts", "author": "@geopolitics_now",
             "date": "2025-11-09",
             "content": "RT DefenseLeaks. Hearing similar from a second source. Threading."},
            {"n": 3, "source": "twitter_posts", "author": "@me_intel",
             "date": "2025-11-10",
             "content": "DefenseLeaks chain has now been picked up by 12 mid-tier OSINT accounts. None named a primary source."},
            {"n": 4, "source": "facebook_posts", "author": "MediaCheck",
             "date": "2025-11-10",
             "content": "Reminder: DefenseLeaks has a history of unverified strike rumors, three of which never materialized."},
        ],
        "answer": (
            "The originating channel is DefenseLeaks on Telegram, claiming sources confirm a "
            "strike package and to 'watch the next 72 hours' [1]. @geopolitics_now is the primary "
            "Twitter amplifier, retweeting DefenseLeaks and saying a second source corroborates "
            "[2]. @me_intel notes that 12 mid-tier OSINT accounts have since picked up the chain, "
            "none naming a primary source [3], and MediaCheck cautions that DefenseLeaks has a "
            "history of three unmaterialized strike rumors [4]."
        ),
    },
    {
        "question": "אילו חשבונות מקדמים את הקריאה להחלפת הממשלה?",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@nehama_il",
             "date": "2026-04-12",
             "content": "ממשלה זו איבדה את הלגיטימציה. דורשים בחירות מיידיות. #החליפו_עכשיו"},
            {"n": 2, "source": "telegram_posts", "author": "המטה האזרחי",
             "date": "2026-04-12",
             "content": "מצרפים את הקריאה של @nehama_il. כל ערוץ – הפיצו. סיסמה אחידה: 'להחליף עכשיו'."},
            {"n": 3, "source": "facebook_posts", "author": "ניטור הרשת",
             "date": "2026-04-13",
             "content": "‏#החליפו_עכשיו פעיל ב-180 חשבונות שנפתחו במרץ – זרימה מתואמת ולא אורגנית."},
            {"n": 4, "source": "twitter_posts", "author": "@yossi_b_ri",
             "date": "2026-04-12",
             "content": "אם זו הקואליציה, לכו הביתה. רק כך נציל את המדינה."},
        ],
        "answer": (
            "החשבון @nehama_il הוא אחד הקולות המובילים, וטוען כי הממשלה איבדה לגיטימציה "
            "ודורש בחירות מיידיות תחת ההאשטג #החליפו_עכשיו [1]. 'המטה האזרחי' בטלגרם מאמץ את "
            "הקריאה ומורה לערוצים להפיץ סיסמה אחידה [2]. 'ניטור הרשת' מזהה דפוס לא אורגני: "
            "‏#החליפו_עכשיו פעיל ב-180 חשבונות שנפתחו במרץ [3]. @yossi_b_ri מצטרף בקריאה "
            "לפיזור הקואליציה [4]."
        ),
    },
    {
        "question": "מי הם המפיצים המרכזיים של תיאוריית הקונספירציה הזו?",
        "posts": [
            {"n": 1, "source": "telegram_posts", "author": "האמת מאחורי הכותרות",
             "date": "2025-07-04",
             "content": "הם לא רוצים שתדעו: ועד מנהל סודי שולט בכל הבנקים. נחשפו מסמכים."},
            {"n": 2, "source": "twitter_posts", "author": "@matteo_il",
             "date": "2025-07-04",
             "content": "ההוכחות מתחילות להיערם. מי שעדיין מאמין לתקשורת המסורתית שיתעורר."},
            {"n": 3, "source": "facebook_posts", "author": "אמת לעם",
             "date": "2025-07-05",
             "content": "מצרפים את הסקירה של 'האמת מאחורי הכותרות'. שתפו, אל תתנו לזה להישכח."},
            {"n": 4, "source": "telegram_posts", "author": "מבדק עובדות IL",
             "date": "2025-07-05",
             "content": "המסמכים שמופצים זויפו ב-Photoshop. הצבע השונה בכותרת מסגיר את העריכה."},
        ],
        "answer": (
            "המפיץ הראשי הוא ערוץ הטלגרם 'האמת מאחורי הכותרות' המפרסם את 'מסמכי' הוועד הסודי "
            "[1]. בטוויטר @matteo_il מקדם את התיאוריה כעובדה ומזלזל בתקשורת המסורתית [2], "
            "ובפייסבוק 'אמת לעם' משתף את הסקירה ומבקש להפיצה הלאה [3]. 'מבדק עובדות IL' מציין "
            "כי המסמכים זויפו ב-Photoshop ושינוי גוון בכותרת חושף את העריכה [4]."
        ),
    },
    {
        "question": "ما هي الحسابات التي تعيد نشر منشورات حركة 'الفجر الجديد'؟",
        "posts": [
            {"n": 1, "source": "telegram_posts", "author": "الفجر الجديد - الرسمية",
             "date": "2025-05-22",
             "content": "بياننا الأول: نرفض المسار الحالي وندعو لتشكيل جبهة شعبية. شاركوا بياننا."},
            {"n": 2, "source": "twitter_posts", "author": "@sawt_alhaq",
             "date": "2025-05-22",
             "content": "إعادة نشر بيان 'الفجر الجديد'. هذا هو الصوت الذي ننتظره."},
            {"n": 3, "source": "facebook_posts", "author": "شباب الوحدة",
             "date": "2025-05-23",
             "content": "نشاركم بيان حركة الفجر الجديد ونؤيد مطالبها. الجبهة الشعبية واجب الوقت."},
            {"n": 4, "source": "twitter_posts", "author": "@mediawatch_ar",
             "date": "2025-05-23",
             "content": "12 حساباً جديداً نشأ هذا الشهر يعيد نشر بيانات الفجر الجديد بنفس الصياغة بالضبط."},
        ],
        "answer": (
            "القناة الأصلية هي 'الفجر الجديد - الرسمية' على تيليغرام التي نشرت البيان الأول "
            "للحركة [1]. الحساب @sawt_alhaq على تويتر يعيد نشر البيان ويؤيده [2]، وصفحة 'شباب "
            "الوحدة' على فيسبوك تتبنى المطالب أيضاً [3]. @mediawatch_ar يلاحظ سلوكاً منسقاً: "
            "12 حساباً نشأ هذا الشهر يعيد نشر البيانات بنفس الصياغة [4]."
        ),
    },

    # ============================================================
    # 6. EVENT TIMELINE / CHRONOLOGY
    # ============================================================
    {
        "question": "When did the rumor about the missing aid convoy first appear?",
        "posts": [
            {"n": 1, "source": "telegram_posts", "author": "Field Reporter A",
             "date": "2025-10-14",
             "content": "Hearing reports of an aid convoy that didn't reach destination this morning. Awaiting confirmation."},
            {"n": 2, "source": "twitter_posts", "author": "@aid_watch",
             "date": "2025-10-14",
             "content": "Picking up the same reports as Field Reporter A. Trucks left at 04:30, never arrived at the warehouse."},
            {"n": 3, "source": "facebook_posts", "author": "Aid Coalition",
             "date": "2025-10-15",
             "content": "Yesterday's missing-convoy reports turned out partly true: 4 of 12 trucks were diverted at a checkpoint."},
            {"n": 4, "source": "twitter_posts", "author": "@official_aid_org",
             "date": "2025-10-15",
             "content": "Official statement: 8 trucks delivered, 4 held for inspection, all crews safe. Investigation underway."},
        ],
        "answer": (
            "The earliest mention is Field Reporter A on Telegram on Oct 14, reporting an aid "
            "convoy that did not reach destination that morning and awaiting confirmation [1]. "
            "@aid_watch picked up the same reports later that day, adding that trucks left at "
            "04:30 and never arrived at the warehouse [2]. By Oct 15 the Aid Coalition partially "
            "confirmed it — 4 of 12 trucks were diverted at a checkpoint [3] — and "
            "@official_aid_org issued an official statement: 8 delivered, 4 held for inspection, "
            "all crews safe [4]."
        ),
    },
    {
        "question": "What was the sequence of posts about the explosion at the central market?",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@first_responder_il",
             "date": "2026-04-22",
             "content": "Loud explosion just now near the spice section. Smoke, sirens. Stay away."},
            {"n": 2, "source": "telegram_posts", "author": "Police Live",
             "date": "2026-04-22",
             "content": "12:14 — large explosion reported at central market. Forces and ambulances dispatched."},
            {"n": 3, "source": "facebook_posts", "author": "City Hall",
             "date": "2026-04-22",
             "content": "12:55 — preliminary: gas-cylinder failure in a stall. 3 injured, no fatalities. Market closed for the day."},
            {"n": 4, "source": "twitter_posts", "author": "@market_insider",
             "date": "2026-04-22",
             "content": "Was inside when it happened. Stall #44 caught fire after a hissing sound. Vendor evacuated everyone."},
        ],
        "answer": (
            "@first_responder_il posted the initial eyewitness report — a loud explosion near the "
            "spice section with smoke and sirens [1]. About the same time, Police Live logged it "
            "officially at 12:14 and dispatched forces and ambulances [2]. By 12:55 City Hall "
            "released a preliminary cause (gas-cylinder failure in a stall, 3 injured, no "
            "fatalities, market closed) [3]. @market_insider added on-the-ground detail: stall #44 "
            "caught fire after a hissing sound and the vendor evacuated everyone [4]."
        ),
    },
    {
        "question": "متى ظهرت أول إشاعة عن استقالة الوزير؟",
        "posts": [
            {"n": 1, "source": "telegram_posts", "author": "همس السياسة",
             "date": "2026-03-01",
             "content": "مصادر تتحدث عن نية وزير المالية تقديم استقالته خلال 48 ساعة. لا تأكيد رسمي."},
            {"n": 2, "source": "twitter_posts", "author": "@daily_brief_ar",
             "date": "2026-03-01",
             "content": "نقلاً عن همس السياسة: الوزير قد يستقيل قريباً. ننتظر تعليقاً رسمياً."},
            {"n": 3, "source": "facebook_posts", "author": "الديوان الإعلامي",
             "date": "2026-03-02",
             "content": "نفي قاطع: الوزير لم يقدم ولن يقدم استقالته. ما يُتداول إشاعات لا أساس لها."},
            {"n": 4, "source": "twitter_posts", "author": "@analyst_x",
             "date": "2026-03-02",
             "content": "النفي الرسمي لا يحسم شيئاً. عادةً ما يسبق نفي كهذا الإعلان بأيام."},
        ],
        "answer": (
            "ظهرت الإشاعة لأول مرة على قناة 'همس السياسة' في 1 مارس، بادعاء أن وزير المالية "
            "ينوي تقديم استقالته خلال 48 ساعة دون تأكيد رسمي [1]. @daily_brief_ar نقلها "
            "في اليوم نفسه بانتظار تعليق رسمي [2]. في 2 مارس نفى 'الديوان الإعلامي' الأمر "
            "نفياً قاطعاً [3]، فيما رأى @analyst_x أن النفي الرسمي لا يحسم شيئاً وقد يسبق "
            "الإعلان بأيام [4]."
        ),
    },
    {
        "question": "ما تسلسل المنشورات حول حادث التفجير في السوق؟",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@news_first",
             "date": "2025-09-18",
             "content": "10:42 - انفجار قوي قرب مدخل السوق الرئيسي. أصوات صفارات إنذار. تجنبوا المنطقة."},
            {"n": 2, "source": "telegram_posts", "author": "غرفة الطوارئ",
             "date": "2025-09-18",
             "content": "10:55 - تأكيد: انفجار في السوق، جرحى وحالة طارئة، الإسعاف في الموقع."},
            {"n": 3, "source": "facebook_posts", "author": "بلدية المدينة",
             "date": "2025-09-18",
             "content": "12:00 - الحصيلة الأولية: 5 جرحى بإصابات متوسطة، السبب قيد التحقيق، السوق مغلق."},
            {"n": 4, "source": "twitter_posts", "author": "@witness_today",
             "date": "2025-09-18",
             "content": "كنت على بعد 50 متراً. سمعت دوياً ثم رأيت دخاناً أبيض. ظننا أنها أسطوانة غاز."},
        ],
        "answer": (
            "بدأت التغطية الساعة 10:42 مع تغريدة @news_first عن انفجار قوي قرب مدخل السوق "
            "وأصوات صفارات إنذار [1]. الساعة 10:55 أكدت 'غرفة الطوارئ' وقوع انفجار وجرحى "
            "ووجود الإسعاف في الموقع [2]. الساعة 12:00 أعلنت 'بلدية المدينة' حصيلة أولية: "
            "5 جرحى بإصابات متوسطة والسبب قيد التحقيق [3]. أضاف الشاهد @witness_today أنه "
            "سمع دوياً ثم رأى دخاناً أبيض وظن أنها أسطوانة غاز [4]."
        ),
    },
    {
        "question": "מתי הופיעה לראשונה השמועה על מבצע צבאי בצפון?",
        "posts": [
            {"n": 1, "source": "telegram_posts", "author": "מקור צבאי",
             "date": "2026-01-15",
             "content": "נשמעות הכנות לפעולה רחבה בגזרה הצפונית. שווה לעקוב בימים הקרובים."},
            {"n": 2, "source": "twitter_posts", "author": "@defense_il",
             "date": "2026-01-16",
             "content": "מצטטים את 'מקור צבאי' מאתמול. לא ראינו עדיין אישור עצמאי."},
            {"n": 3, "source": "facebook_posts", "author": "דובר צה\"ל הרשמי",
             "date": "2026-01-16",
             "content": "אין שינוי במצב המבצעי. שמועות על מבצע מתוכנן אינן נכונות."},
            {"n": 4, "source": "telegram_posts", "author": "ניתוח שטח",
             "date": "2026-01-17",
             "content": "אישור הכחשה רשמית. בכל זאת ראינו ניוד ניכר של רכבי מילואים בליל ה-15."},
        ],
        "answer": (
            "השמועה הופיעה לראשונה ב-15 בינואר בערוץ הטלגרם 'מקור צבאי', שדיווח על הכנות "
            "לפעולה רחבה בגזרה הצפונית [1]. למחרת @defense_il ציטט אותה ללא אישור עצמאי [2]. "
            "באותו יום פרסם דובר צה\"ל הכחשה רשמית [3], אך 'ניתוח שטח' מציין כי למרות "
            "ההכחשה ניצפה ניוד ניכר של רכבי מילואים בליל ה-15 [4]."
        ),
    },

    # ============================================================
    # 7. GEOGRAPHIC
    # ============================================================
    {
        "question": "What's happening in Gaza border communities right now?",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@otef_news",
             "date": "2025-12-10",
             "content": "Sirens in Nahal Oz and Kfar Aza in the last hour. Residents in shelters."},
            {"n": 2, "source": "facebook_posts", "author": "Sderot Live",
             "date": "2025-12-10",
             "content": "School day cancelled across the regional council. Buses turned back at 07:30."},
            {"n": 3, "source": "telegram_posts", "author": "Border Watch",
             "date": "2025-12-10",
             "content": "IDF reports two interceptions over the regional area. No injuries reported. Roads partly closed."},
        ],
        "answer": (
            "@otef_news reports sirens in Nahal Oz and Kfar Aza within the last hour, with "
            "residents in shelters [1]. Sderot Live notes that the regional school day has been "
            "cancelled and buses were turned back at 07:30 [2]. Border Watch adds that the IDF "
            "reported two interceptions over the regional area, no injuries, and partially closed "
            "roads [3]."
        ),
    },
    {
        "question": "Any incidents reported in northern Israel today?",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@north_alerts",
             "date": "2026-02-14",
             "content": "Anti-tank fire warning earlier in Margaliot. All clear after 25 minutes."},
            {"n": 2, "source": "telegram_posts", "author": "Galilee Live",
             "date": "2026-02-14",
             "content": "Drone alert near Shtula at 14:10. Forces deployed, no casualties so far."},
            {"n": 3, "source": "facebook_posts", "author": "Kiryat Shmona Today",
             "date": "2026-02-14",
             "content": "Quiet evening here so far. Cafes open again on the main street after lunchtime warning."},
            {"n": 4, "source": "instagram_posts", "author": "tlv_runner",
             "date": "2026-02-14",
             "content": "10 km along the beach. Best run of the month."},
        ],
        "answer": (
            "@north_alerts reports an anti-tank fire warning earlier in Margaliot that was cleared "
            "after 25 minutes [1]. Galilee Live logs a drone alert near Shtula at 14:10 with "
            "forces deployed and no casualties so far [2]. Kiryat Shmona Today describes a quiet "
            "evening with cafes reopening after a lunchtime warning [3]. Post [4] is unrelated "
            "(running content from Tel Aviv)."
        ),
    },
    {
        "question": "מה קורה ביישובי עוטף עזה הבוקר?",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@otef_now",
             "date": "2025-11-19",
             "content": "אזעקות בנחל עוז ובארי ב-06:40. תושבים במרחבים מוגנים."},
            {"n": 2, "source": "facebook_posts", "author": "מועצה אזורית שער הנגב",
             "date": "2025-11-19",
             "content": "ביטול לימודים בכל המוסדות החינוכיים היום. הסעות לא יוצאות."},
            {"n": 3, "source": "telegram_posts", "author": "מבט לדרום",
             "date": "2025-11-19",
             "content": "צה\"ל מאשר יירוט מעל האזור. אין נפגעים. כביש 232 חסום בקטע אחד."},
            {"n": 4, "source": "instagram_posts", "author": "ofakim.life",
             "date": "2025-11-19",
             "content": "אופקים מתעוררת. ילדים בגינה ליד בית הכנסת. הבוקר רגוע יחסית כאן."},
        ],
        "answer": (
            "‏@otef_now מדווח על אזעקות בנחל עוז ובארי ב-06:40 ועל תושבים במרחבים מוגנים [1]. "
            "המועצה האזורית שער הנגב הודיעה על ביטול לימודים בכל המוסדות והסעות שאינן יוצאות "
            "[2]. 'מבט לדרום' מאשר יירוט מעל האזור ללא נפגעים, וקטע מכביש 232 חסום [3]. "
            "באופקים, מעט רחוק יותר, התמונה רגועה יחסית – ilds בגינה ליד בית הכנסת [4]."
        ),
    },
    {
        "question": "אילו דיווחים יש מהצפון בשעות האחרונות?",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@galil_alert",
             "date": "2025-08-30",
             "content": "התרעת חדירת כלי טיס במרגליות לפני כ-40 דקות. אין נפגעים."},
            {"n": 2, "source": "telegram_posts", "author": "צפון עכשיו",
             "date": "2025-08-30",
             "content": "ירי נגד טנקים זוהה בכפר יובל. הכוחות בשטח. תושבים במרחבים."},
            {"n": 3, "source": "facebook_posts", "author": "קריית שמונה היום",
             "date": "2025-08-30",
             "content": "בעיר עצמה: שקט. בעלי עסקים פתחו מחדש. אנשים יושבים בקפה ליד הכיכר."},
        ],
        "answer": (
            "‏@galil_alert מדווח על התרעת חדירת כלי טיס במרגליות לפני כ-40 דקות, ללא נפגעים [1]. "
            "'צפון עכשיו' מציין כי בכפר יובל זוהה ירי נגד טנקים, הכוחות בשטח והתושבים במרחבים "
            "מוגנים [2]. בקריית שמונה הדיווח שונה: 'קריית שמונה היום' מתאר שקט יחסי, בעלי "
            "עסקים שפתחו מחדש ואנשים בבית הקפה ליד הכיכר [3]."
        ),
    },
    {
        "question": "ما الذي يحدث في مدينة الخليل اليوم؟",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@hebron_news",
             "date": "2026-02-28",
             "content": "اشتباكات في حي تل رميدة منذ ساعة. غاز مسيل للدموع وحجارة. لا إصابات بالغة حتى الآن."},
            {"n": 2, "source": "telegram_posts", "author": "نبض الخليل",
             "date": "2026-02-28",
             "content": "إغلاق طريق وسط البلدة القديمة. حركة السير متوقفة عند الحاجز الجنوبي."},
            {"n": 3, "source": "facebook_posts", "author": "بلدية الخليل",
             "date": "2026-02-28",
             "content": "تعليق دوام المدارس في البلدة القديمة بعد ظهر اليوم. ندعو الأهالي إلى الحذر."},
        ],
        "answer": (
            "@hebron_news يفيد بوقوع اشتباكات في حي تل رميدة منذ ساعة، مع غاز مسيل للدموع "
            "وحجارة ودون إصابات بالغة حتى الآن [1]. 'نبض الخليل' يضيف أن طريق وسط البلدة "
            "القديمة مغلق وحركة السير متوقفة عند الحاجز الجنوبي [2]. 'بلدية الخليل' علقت دوام "
            "المدارس في البلدة القديمة بعد الظهر ودعت الأهالي إلى الحذر [3]."
        ),
    },

    # ============================================================
    # 8. DISINFORMATION / CLAIM VERIFICATION
    # ============================================================
    {
        "question": "Is there fake news circulating about the IDF spokesperson?",
        "posts": [
            {"n": 1, "source": "telegram_posts", "author": "Inside Story",
             "date": "2025-10-25",
             "content": "Spokesperson resigned this morning. Internal sources confirm it. Official statement coming soon."},
            {"n": 2, "source": "twitter_posts", "author": "@idf_press",
             "date": "2025-10-25",
             "content": "Reports of a resignation are false. Spokesperson is on duty and will brief reporters at 18:00."},
            {"n": 3, "source": "facebook_posts", "author": "FactCheck IL",
             "date": "2025-10-25",
             "content": "The 'resignation' image circulating is a doctored screenshot. Original post about a routine schedule change."},
            {"n": 4, "source": "twitter_posts", "author": "@news_basic",
             "date": "2025-10-25",
             "content": "Confused timeline today. Two outlets ran the resignation story before retracting within an hour."},
        ],
        "answer": (
            "Yes — Inside Story on Telegram circulated a claim that the spokesperson resigned this "
            "morning, citing internal sources [1]. The official @idf_press account explicitly "
            "denied this, saying the spokesperson is on duty and will brief reporters at 18:00 [2]. "
            "FactCheck IL identifies the viral 'resignation' image as a doctored screenshot of an "
            "original post about a routine schedule change [3]. @news_basic adds that two outlets "
            "ran and retracted the story within an hour [4]."
        ),
    },
    {
        "question": "Are these claims about a hospital strike credible?",
        "posts": [
            {"n": 1, "source": "telegram_posts", "author": "War Room Live",
             "date": "2025-12-18",
             "content": "Direct strike on Al-Shams hospital this morning. Heavy casualties. Photos coming."},
            {"n": 2, "source": "twitter_posts", "author": "@osint_geo",
             "date": "2025-12-18",
             "content": "The first photo posted is from a 2022 incident in a different city. Reverse image search confirms."},
            {"n": 3, "source": "facebook_posts", "author": "Hospital Press Office",
             "date": "2025-12-18",
             "content": "There was no strike on our facility today. Operations are normal. We are aware of false reports."},
            {"n": 4, "source": "twitter_posts", "author": "@field_journalist",
             "date": "2025-12-18",
             "content": "Verified on the ground: hospital intact, staff working normally. Original claim appears fabricated."},
        ],
        "answer": (
            "The claim does not appear credible. War Room Live alleged a direct strike on "
            "Al-Shams hospital with heavy casualties [1], but @osint_geo found that the first "
            "photo used was from a 2022 incident in a different city, confirmed via reverse image "
            "search [2]. The Hospital Press Office officially denies any strike and says "
            "operations are normal [3], and @field_journalist verified on the ground that the "
            "hospital is intact and staff are working normally [4]."
        ),
    },
    {
        "question": "هل تنتشر أخبار مزيفة عن وفاة الوزير؟",
        "posts": [
            {"n": 1, "source": "telegram_posts", "author": "أخبار سريعة",
             "date": "2026-01-22",
             "content": "عاجل: وفاة وزير الداخلية بسكتة قلبية. التفاصيل لاحقاً."},
            {"n": 2, "source": "twitter_posts", "author": "@official_gov",
             "date": "2026-01-22",
             "content": "نفي قاطع: الوزير في صحة جيدة وحضر اجتماع مجلس الوزراء قبل ساعة."},
            {"n": 3, "source": "facebook_posts", "author": "تحقق من الأخبار",
             "date": "2026-01-22",
             "content": "الصورة المتداولة للوزير قديمة من 2019 وأُعيد تدويرها مع نص الوفاة الكاذب."},
            {"n": 4, "source": "twitter_posts", "author": "@journalist_ali",
             "date": "2026-01-22",
             "content": "شاهدت الوزير شخصياً يدخل البرلمان قبل ساعتين. الخبر مزيف بالكامل."},
        ],
        "answer": (
            "نعم تنتشر هذه الأخبار وهي مزيفة. قناة 'أخبار سريعة' نشرت خبراً عاجلاً عن وفاة "
            "الوزير بسكتة قلبية [1]، لكن @official_gov نفى ذلك قاطعاً وأكد حضوره اجتماع مجلس "
            "الوزراء قبل ساعة [2]. 'تحقق من الأخبار' يكشف أن الصورة المتداولة قديمة من 2019 "
            "أُعيد تدويرها مع نص كاذب [3]، و@journalist_ali شاهد الوزير شخصياً يدخل البرلمان "
            "قبل ساعتين [4]."
        ),
    },
    {
        "question": "هل صحيح أن الجيش أعلن عن هدنة جديدة؟",
        "posts": [
            {"n": 1, "source": "telegram_posts", "author": "أخبار الجبهة",
             "date": "2026-03-08",
             "content": "مصادرنا تؤكد إعلان هدنة لمدة 48 ساعة بدءاً من منتصف الليل. الانتظار للبيان الرسمي."},
            {"n": 2, "source": "twitter_posts", "author": "@army_press_ar",
             "date": "2026-03-08",
             "content": "لم يصدر عنا أي إعلان عن هدنة. الادعاءات المتداولة لا أساس لها."},
            {"n": 3, "source": "facebook_posts", "author": "تحقق من الأخبار",
             "date": "2026-03-08",
             "content": "بيان الهدنة المتداول مزور: شعار الناطق فيه يختلف عن الأصلي بحرفين."},
        ],
        "answer": (
            "لا، الادعاء غير صحيح. 'أخبار الجبهة' روّجت لإعلان هدنة لمدة 48 ساعة بدءاً من "
            "منتصف الليل بانتظار بيان رسمي [1]، إلا أن @army_press_ar نفى صدور أي إعلان "
            "ووصف الادعاءات بأنها بلا أساس [2]. كذلك يوضح 'تحقق من الأخبار' أن البيان "
            "المتداول مزور وأن شعار الناطق فيه يختلف عن الأصلي بحرفين [3]."
        ),
    },
    {
        "question": "האם מסתובבת פייק ניוז על דובר המשטרה?",
        "posts": [
            {"n": 1, "source": "telegram_posts", "author": "חדשות עכשיו",
             "date": "2025-09-05",
             "content": "דובר המשטרה הושעה הבוקר על רקע פרשה פלילית. מצפים לאישור."},
            {"n": 2, "source": "twitter_posts", "author": "@police_press_il",
             "date": "2025-09-05",
             "content": "ההודעה על השעיית הדובר אינה נכונה. הוא בעבודה ויקיים תדרוך הערב."},
            {"n": 3, "source": "facebook_posts", "author": "מבדק עובדות IL",
             "date": "2025-09-05",
             "content": "הצילום שהופץ הוא מסמך משנת 2021 שעבר עיבוד גרפי. הקישור לפרשה הנוכחית מומצא."},
        ],
        "answer": (
            "כן – ערוץ 'חדשות עכשיו' בטלגרם הפיץ ידיעה לפיה דובר המשטרה הושעה הבוקר על רקע "
            "פרשה פלילית [1]. החשבון הרשמי @police_press_il הכחיש זאת מפורשות וציין שהדובר "
            "בעבודה ויקיים תדרוך הערב [2]. 'מבדק עובדות IL' מצביע על כך שהצילום שהופץ הוא "
            "מסמך משנת 2021 שעבר עיבוד גרפי, ושהקישור לפרשה הנוכחית מומצא [3]."
        ),
    },

    # ============================================================
    # 9. COMPARATIVE
    # ============================================================
    {
        "question": "Compare the Hebrew vs Arabic discourse on the hostage deal.",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@israel_voices",
             "date": "2026-03-25",
             "content": "Hebrew feed today: 'any deal worth saving lives', families' cries leading the discourse."},
            {"n": 2, "source": "facebook_posts", "author": "סקירת רשתות",
             "date": "2026-03-25",
             "content": "השיח בעברית מתמקד במשפחות החטופים ובמחיר הביטחוני של עסקה. פחות עיסוק במה שמשתחרר בצד השני."},
            {"n": 3, "source": "twitter_posts", "author": "@arab_pulse",
             "date": "2026-03-25",
             "content": "השיח הערבי: דגש על שמות האסירים שישוחררו ועל סמליות הרגע. עסקה כניצחון פוליטי."},
            {"n": 4, "source": "facebook_posts", "author": "مرصد الإعلام",
             "date": "2026-03-25",
             "content": "النقاش بالعربية يتمحور حول الأسرى الذين سيُفرج عنهم وعدد العائدين. الخطاب احتفالي إلى حد كبير."},
        ],
        "answer": (
            "The Hebrew discourse centers on the hostage families and the security cost of a "
            "deal — @israel_voices summarizes it as 'any deal worth saving lives' [1] and 'סקירת "
            "רשתות' notes less attention to who is released on the other side [2]. The Arabic "
            "discourse centers on the prisoners to be freed and the symbolism of the moment, "
            "framed as a political victory by @arab_pulse [3], with 'مرصد الإعلام' describing "
            "the tone as largely celebratory and focused on the names and number of returnees [4]."
        ),
    },
    {
        "question": "How does the Twitter discussion differ from Telegram about the protest?",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@protest_watch",
             "date": "2026-04-30",
             "content": "On Twitter the protest is debated openly, lots of disagreement and counter-voices in replies."},
            {"n": 2, "source": "telegram_posts", "author": "Mobilize Now",
             "date": "2026-04-30",
             "content": "Saturday at 19:00, Kaplan. Bring water and signs. No counter-voices, just logistics."},
            {"n": 3, "source": "twitter_posts", "author": "@civic_index",
             "date": "2026-04-30",
             "content": "Twitter: ~55% supportive, ~30% opposed, ~15% neutral analysis. Telegram channels are >95% supportive."},
            {"n": 4, "source": "telegram_posts", "author": "Field Coordinators",
             "date": "2026-04-30",
             "content": "Marshals meeting at 17:30 near the lions. Wear yellow vests. Coordinated routes attached."},
        ],
        "answer": (
            "On Twitter the discussion is debated and pluralistic — @protest_watch describes lots "
            "of disagreement and counter-voices in replies [1], and @civic_index measures roughly "
            "55% supportive, 30% opposed, 15% neutral [3]. Telegram channels are operational "
            "rather than deliberative: 'Mobilize Now' broadcasts logistics for Saturday 19:00 at "
            "Kaplan with no counter-voices [2], and 'Field Coordinators' organizes a marshals "
            "meeting at 17:30 with vests and routes [4]. @civic_index puts Telegram support above "
            "95% [3]."
        ),
    },
    {
        "question": "השווה את השיח על המלחמה בעברית ובערבית.",
        "posts": [
            {"n": 1, "source": "facebook_posts", "author": "מבט תקשורתי",
             "date": "2025-11-28",
             "content": "השיח העברי השבוע מתמקד באבדות צה\"ל ובחיילי המילואים. דומיננטיות של פוסטי הספד וזיכרון."},
            {"n": 2, "source": "telegram_posts", "author": "ניטור עברית",
             "date": "2025-11-28",
             "content": "סנטימנט עברי: 50% עצב/אבל, 25% תמיכה במבצע, 25% ביקורת על הדרג המדיני."},
            {"n": 3, "source": "twitter_posts", "author": "@arab_monitor",
             "date": "2025-11-28",
             "content": "השיח הערבי השבוע: דגש על התושבים בעזה, תיעוד הפצועים, וביקורת על המתווכים הבינלאומיים."},
            {"n": 4, "source": "facebook_posts", "author": "مرصد العربية",
             "date": "2025-11-28",
             "content": "النقاش بالعربية يركز على الضحايا المدنيين والوضع الإنساني، مع غضب من الموقف الدولي."},
        ],
        "answer": (
            "השיח בעברית מתרכז סביב אובדן חיילים ומילואימניקים, עם פוסטי הספד וזיכרון בולטים "
            "['מבט תקשורתי' [1]], ופילוח של 'ניטור עברית' מצביע על כ-50% אבל, 25% תמיכה "
            "במבצע ו-25% ביקורת על הדרג המדיני [2]. השיח בערבית מתמקד דווקא בתושבי עזה, "
            "בתיעוד הפצועים ובביקורת על המתווכים הבינלאומיים – כך @arab_monitor [3] – ו'مرصد "
            "العربية' מוסיף כי הכעס מופנה גם כלפי המוקד הבינלאומי וכלפי המצב ההומניטרי [4]."
        ),
    },
    {
        "question": "מה ההבדל בין דיווחי פייסבוק לאינסטגרם על האירוע בכיכר?",
        "posts": [
            {"n": 1, "source": "facebook_posts", "author": "תושבי המרכז",
             "date": "2026-05-04",
             "content": "באירוע בכיכר אתמול הופיע ראש העירייה, שני סגניו, ושני חברי מועצה. נאומים על תקציב חינוך."},
            {"n": 2, "source": "facebook_posts", "author": "בלוג עיריית מרכז",
             "date": "2026-05-04",
             "content": "סיכום האירוע: 1500 משתתפים, התחייבות ל-30 מיליון לחינוך, פתיחה של ועדה ציבורית."},
            {"n": 3, "source": "instagram_posts", "author": "tlv_pics",
             "date": "2026-05-04",
             "content": "אווירה מדהימה אתמול בכיכר. מוזיקה, אוכל, ובלונים. שווה לראות את הסטוריס."},
            {"n": 4, "source": "instagram_posts", "author": "noa.shoots",
             "date": "2026-05-04",
             "content": "תמונות צבעוניות מהאירוע. הילדים נהנו, ההורים שתו קפה, הכל נראה חגיגי."},
        ],
        "answer": (
            "פייסבוק מתמקד בצד התוכני והפוליטי: 'תושבי המרכז' מדווח על נוכחות ראש העיר וסגניו "
            "ועל נאומים בנושא תקציב חינוך [1], ו'בלוג עיריית מרכז' מציין 1500 משתתפים, "
            "התחייבות ל-30 מיליון לחינוך והקמת ועדה ציבורית [2]. אינסטגרם, לעומת זאת, מתמקד "
            "באווירה ובוויזואל: tlv_pics מתאר מוזיקה ואוכל ובלונים [3], ו-noa.shoots מציגה "
            "תמונות צבעוניות של ילדים נהנים והורים שתו קפה [4]."
        ),
    },
    {
        "question": "قارن بين الخطاب العبري والعربي حول صفقة الأسرى.",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@hebrew_pulse",
             "date": "2025-09-12",
             "content": "השיח העברי: מחיר ביטחוני, חרדה לחיי החטופים, פחות הצגה של מי משתחרר בצד השני."},
            {"n": 2, "source": "facebook_posts", "author": "ניטור שיח",
             "date": "2025-09-12",
             "content": "השפה העברית מציגה את העסקה דרך הפריזמה של חזרה הביתה, כמעט ולא דרך הצד הפלסטיני."},
            {"n": 3, "source": "twitter_posts", "author": "@arab_voice",
             "date": "2025-09-12",
             "content": "النقاش العربي يركز على أسماء الأسرى ولحظة العودة. خطاب احتفالي ووحدوي."},
            {"n": 4, "source": "facebook_posts", "author": "مرصد الخطاب",
             "date": "2025-09-12",
             "content": "اللغة العربية تتعامل مع الصفقة كانتصار رمزي. تركيز على الأسر والعائلات المستقبِلة."},
        ],
        "answer": (
            "الخطاب العبري يركز على الكلفة الأمنية والقلق على حياة الرهائن مع تقليل الإشارة "
            "إلى من سيُفرج عنه في المقابل، كما يلخّص @hebrew_pulse [1] و'ניטור שיח' الذي "
            "يلاحظ أن الإطار السائد هو 'العودة إلى البيت' وليس الجانب الفلسطيني [2]. أما "
            "الخطاب العربي فيتمحور حول أسماء الأسرى ولحظة العودة بنبرة احتفالية ووحدوية وفق "
            "@arab_voice [3]، و'مرصد الخطاب' يصف الصفقة باعتبارها انتصاراً رمزياً يتركز "
            "حول الأسر والعائلات المستقبِلة [4]."
        ),
    },

    # ============================================================
    # 10. POSTS DON'T ANSWER (rule #4)
    # ============================================================
    {
        "question": "What did the prime minister say at the UN General Assembly today?",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@cricket_fans_il",
             "date": "2025-09-22",
             "content": "Match of the season tonight. Blue team needs this win to stay in contention."},
            {"n": 2, "source": "facebook_posts", "author": "Coffee Lovers TLV",
             "date": "2025-09-22",
             "content": "New roastery opened on Frishman. Their cold brew is excellent."},
            {"n": 3, "source": "instagram_posts", "author": "leah.bakes",
             "date": "2025-09-22",
             "content": "Babka with chocolate and tahini. Recipe in stories."},
        ],
        "answer": (
            "The retrieved posts do not address the question. They cover an upcoming cricket "
            "match [1], a new coffee roastery in Tel Aviv [2], and a babka recipe [3] — none "
            "mention the prime minister, the UN, or the speech. No answer can be given from this "
            "evidence."
        ),
    },
    {
        "question": "Who is the new chief of general staff?",
        "posts": [
            {"n": 1, "source": "twitter_posts", "author": "@startup_news_il",
             "date": "2026-01-04",
             "content": "Series B announced for a Tel Aviv cyber-resilience startup. $42M, led by a US fund."},
            {"n": 2, "source": "facebook_posts", "author": "Hiking IL",
             "date": "2026-01-04",
             "content": "Trail of the week: Mount Meron loop. About 4 hours, beautiful views."},
            {"n": 3, "source": "telegram_posts", "author": "Markets Daily",
             "date": "2026-01-04",
             "content": "TA-35 closed +1.2% today. Banks led the gains."},
            {"n": 4, "source": "instagram_posts", "author": "ron.streetfood",
             "date": "2026-01-04",
             "content": "Best sabich in Givatayim, no question."},
        ],
        "answer": (
            "The retrieved posts do not answer this question. They cover a startup funding round "
            "[1], a hiking trail at Mount Meron [2], a stock-market summary [3], and a sabich "
            "review [4]. None mention the chief of general staff or any military appointment, so "
            "no name can be cited from this evidence."
        ),
    },
    {
        "question": "מה אמר נשיא ארה\"ב בישיבת מועצת הביטחון אמש?",
        "posts": [
            {"n": 1, "source": "facebook_posts", "author": "טבע ישראל",
             "date": "2025-12-01",
             "content": "פריחת הכלניות החלה מוקדם השנה. מומלץ לבקר באזור עוטף עזה בסופ\"ש."},
            {"n": 2, "source": "twitter_posts", "author": "@israel_chefs",
             "date": "2025-12-01",
             "content": "מסעדה חדשה בשרונה פתחה היום. תפריט אסיאתי-ים תיכוני, מעניין."},
            {"n": 3, "source": "instagram_posts", "author": "tlv_runner",
             "date": "2025-12-01",
             "content": "ריצה של 12 ק\"מ הבוקר. אוויר חורף יבש, מצוין."},
        ],
        "answer": (
            "הפוסטים שאוחזרו אינם נוגעים לשאלה. הם עוסקים בפריחת הכלניות בעוטף [1], "
            "במסעדה חדשה בשרונה [2] ובריצת בוקר בתל אביב [3]. אף אחד מהם אינו מזכיר את "
            "נשיא ארה\"ב, מועצת הביטחון או דברי הנשיא, ולכן לא ניתן לענות מתוך החומר הזה."
        ),
    },
    {
        "question": "ما الذي قاله الرئيس الأمريكي في خطابه الأخير عن الشرق الأوسط؟",
        "posts": [
            {"n": 1, "source": "instagram_posts", "author": "yara_cooks",
             "date": "2026-02-11",
             "content": "وصفة مقلوبة دجاج بالبهارات السبعة. شاركوها مع العائلة."},
            {"n": 2, "source": "twitter_posts", "author": "@books_arabic",
             "date": "2026-02-11",
             "content": "صدرت اليوم الترجمة العربية لرواية يابانية مهمة. مراجعة قريباً."},
            {"n": 3, "source": "telegram_posts", "author": "أخبار الرياضة",
             "date": "2026-02-11",
             "content": "نتائج مباريات اليوم: الأهلي 2 - 1 الزمالك، الترجي 3 - 0 النجم."},
            {"n": 4, "source": "facebook_posts", "author": "هواة الفلك",
             "date": "2026-02-11",
             "content": "كسوف جزئي للقمر يمكن مشاهدته في معظم الدول العربية ليلة الجمعة."},
        ],
        "answer": (
            "المنشورات المُسترجعة لا تتطرق للسؤال. فهي تتناول وصفة مقلوبة [1]، وصدور ترجمة "
            "عربية لرواية يابانية [2]، ونتائج مباريات كرة القدم [3]، وكسوف القمر الجزئي [4]، "
            "ولا يذكر أيٌّ منها الرئيس الأمريكي أو الشرق الأوسط أو أي خطاب. لا يمكن استخلاص "
            "إجابة من هذا المحتوى."
        ),
    },
    {
        "question": "كم عدد الجرحى في الانفجار الأخير في وسط المدينة؟",
        "posts": [
            {"n": 1, "source": "facebook_posts", "author": "نادي القراءة",
             "date": "2025-08-08",
             "content": "اجتماع نادي القراءة الأسبوع القادم لمناقشة رواية 'موسم الهجرة إلى الشمال'."},
            {"n": 2, "source": "twitter_posts", "author": "@tech_arab",
             "date": "2025-08-08",
             "content": "إطلاق هاتف جديد اليوم بمواصفات منافسة وسعر معقول."},
            {"n": 3, "source": "instagram_posts", "author": "amman_streets",
             "date": "2025-08-08",
             "content": "صور من شارع الرينبو في عمّان. الحياة الليلية رائعة هذا الصيف."},
        ],
        "answer": (
            "المنشورات المُسترجعة لا تجيب على هذا السؤال. فهي تتحدث عن اجتماع نادي قراءة [1]، "
            "وإطلاق هاتف جديد [2]، وصور من شارع الرينبو في عمّان [3]، ولا يذكر أي منها "
            "انفجاراً أو جرحى. لا يمكن تحديد عدد الجرحى بناءً على هذه المنشورات."
        ),
    },
]


def format_few_shot_block() -> str:
    """Render all examples as a single text block ready to inject into a prompt.

    Mirrors the shape of `format_context` in query.py so the few-shot examples
    look identical to the real retrieved context the LLM sees at inference
    time. The `score=` segment is dropped because these are synthetic.
    """
    blocks: list[str] = []
    for ex in FEW_SHOT_QA_EXAMPLES:
        posts_text_parts: list[str] = []
        for p in ex["posts"]:
            head = (
                f"[{p['n']}] {p['source']} #{p['n']} | {p['author']} | {p['date']}"
            )
            posts_text_parts.append(head + "\n" + (p["content"] or "").strip())
        posts_block = "\n\n".join(posts_text_parts)
        blocks.append(
            f"QUESTION: {ex['question']}\n"
            f"POSTS:\n{posts_block}\n"
            f"ANSWER: {ex['answer']}"
        )
    return "\n\n---\n\n".join(blocks)


if __name__ == "__main__":
    print(f"Loaded {len(FEW_SHOT_QA_EXAMPLES)} few-shot Q&A examples.")
