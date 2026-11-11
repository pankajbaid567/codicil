# Hinglish and Hindi renderings of questions the indexed Directions do not answer. Each inherits the split of its English source (no dev/test leakage).
U = "unanswerable"
H, D = "hinglish", "hi"
ITEMS = [
    ("ux01", "RBI ka abhi ka repo rate kitna hai?", None, None, U, H, "u01"),
    ("ux02", "senior citizens ko fixed deposit ke interest par income tax kitna lagta hai?", None, None, U, H, "u02"),
    ("ux03", "cheque bounce hone par NI Act ki dhara 138 ke tahat kya saza hai?", None, None, U, H, "u03"),
    ("ux04", "India mein kaun se cryptocurrency exchange legal hain?", None, None, U, H, "u04"),
    ("ux05", "Public Provident Fund par kitna byaj milta hai?", None, None, U, H, "u06"),
    ("ux06", "Basel III ke tahat banks ko kitna capital adequacy ratio rakhna padta hai?", None, None, U, H, "u08"),
    ("ux07", "banks ko RBI ke paas abhi kitna CRR rakhna hota hai?", None, None, U, H, "u09"),
    ("ux08", "loan ko NPA kab classify kiya jata hai?", None, None, U, H, "u10"),
    ("ux09", "IMPS se ek din mein maximum kitna transfer ho sakta hai?", None, None, U, H, "u13"),
    ("ux10", "demat account kholne ke kya rules hain?", None, None, U, H, "u14"),
    ("ux11", "bank service fees par kitna GST lagta hai?", None, None, U, H, "u16"),
    ("ux12", "NEFT transfer beneficiary tak pahunchne mein kitna time leta hai?", None, None, U, H, "u20"),
    ("ud01", "रेपो रेट अभी कितना है?", None, None, U, D, "u01"),
    ("ud02", "भारत में कौन से क्रिप्टोकरेंसी एक्सचेंज वैध हैं?", None, None, U, D, "u04"),
    ("ud03", "पब्लिक प्रोविडेंट फंड पर कितना ब्याज मिलता है?", None, None, U, D, "u06"),
    ("ud04", "बैंकों को आरबीआई के पास कितना नकद आरक्षित अनुपात रखना होता है?", None, None, U, D, "u09"),
    ("ud05", "किसी ऋण को अनर्जक आस्ति कब घोषित किया जाता है?", None, None, U, D, "u10"),
    ("ud06", "डीमैट खाता खोलने के क्या नियम हैं?", None, None, U, D, "u14"),
    ("ud07", "बैंक की सेवा शुल्क पर कितना जीएसटी लगता है?", None, None, U, D, "u16"),
    ("ud08", "NEFT ट्रांसफर लाभार्थी तक पहुँचने में कितना समय लेता है?", None, None, U, D, "u20"),
]
