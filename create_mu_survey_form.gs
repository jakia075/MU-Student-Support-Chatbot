/**
 * Creates the user-study Google Form for the MU Student Support Chatbot thesis.
 * How to use: script.google.com -> New project -> paste this code -> Save -> Run "createSurvey".
 * The links of the new form are printed in the Execution log.
 */
const CHATBOT_LINK = "PASTE_THE_NGROK_LINK_HERE";   // change before sharing, or edit the form description later

function createSurvey() {
  const form = FormApp.create("MU Student Support Chatbot — User Evaluation");
  form.setDescription(
    "Thank you for helping us evaluate our thesis chatbot (Dept. of CSE, Metropolitan University, Sylhet).\n" +
    "আমাদের থিসিসের চ্যাটবট মূল্যায়নে সাহায্য করার জন্য ধন্যবাদ।\n\n" +
    "1) Open the chatbot / চ্যাটবট খুলুন: " + CHATBOT_LINK + "\n" +
    "2) Try the tasks below, then answer the questions (about 10 minutes).\n\n" +
    "The survey is anonymous; no e-mail address is collected. / উত্তর সম্পূর্ণ গোপন থাকবে।\n" +
    "Researchers: Jakia Habiba & Mst. Samia Begum · Supervisor: Samia Rahman Rima");
  form.setCollectEmail(false);
  form.setProgressBar(true);
  form.setConfirmationMessage("Thank you! / ধন্যবাদ! Your response has been recorded.");

  // ---------- consent ----------
  form.addMultipleChoiceItem()
    .setTitle("C1. I agree to take part in this anonymous survey. / আমি এই গোপন জরিপে অংশ নিতে সম্মত।")
    .setChoiceValues(["Yes / হ্যাঁ"]).setRequired(true);

  // ---------- background ----------
  form.addPageBreakItem().setTitle("Part 1 — About you / আপনার সম্পর্কে");
  form.addListItem().setTitle("D1. Department / বিভাগ")
    .setChoiceValues(["CSE", "Software Engineering", "Data Science", "EEE", "BBA", "English", "Economics", "LLB", "Other"])
    .setRequired(true);
  form.addListItem().setTitle("D2. Year of study / কোন বর্ষে পড়েন")
    .setChoiceValues(["1st year", "2nd year", "3rd year", "4th year", "Masters"]).setRequired(true);
  form.addMultipleChoiceItem().setTitle("D3. Language you mostly used with the chatbot / চ্যাটবটের সাথে কোন ভাষা বেশি ব্যবহার করেছেন")
    .setChoiceValues(["English", "বাংলা", "Banglish"]).setRequired(true);
  form.addCheckboxItem().setTitle("D4. Where do you usually get university information? / সাধারণত কোথা থেকে বিশ্ববিদ্যালয়ের তথ্য পান")
    .setChoiceValues(["Office visit / অফিসে গিয়ে", "MU website", "Facebook group", "Friends and seniors / বন্ধু ও সিনিয়র", "Teachers / শিক্ষক"])
    .setRequired(true);
  form.addMultipleChoiceItem().setTitle("D5. Have you used an AI chatbot (e.g., ChatGPT) before? / আগে কোনো AI চ্যাটবট ব্যবহার করেছেন?")
    .setChoiceValues(["Yes / হ্যাঁ", "No / না"]).setRequired(true);

  // ---------- tasks ----------
  form.addPageBreakItem().setTitle("Part 2 — Tasks / কাজ")
    .setHelpText("Ask each question in the chatbot, then tell us if the answer was correct. / প্রতিটি প্রশ্ন চ্যাটবটে জিজ্ঞেস করে উত্তর সঠিক ছিল কিনা জানান।");
  const yp = ["Yes / হ্যাঁ", "Partly / আংশিক", "No / না"];
  const tasks = [
    "T1. (English) Ask: \"How many credits are needed for BSc in CSE?\" — Was the answer correct?",
    "T2. (বাংলা) জিজ্ঞেস করুন: \"সাপ্লিমেন্টারি পরীক্ষার রেজিস্ট্রেশন কোথায় করব?\" — উত্তর সঠিক ছিল, বা সঠিক অফিসে পাঠিয়েছে?",
    "T3. (Banglish) Ask: \"retake fee koto?\" — Was the answer correct?",
    "T4. Ask your own question about MU, in any language. Was it answered correctly? / নিজের যেকোনো প্রশ্ন করুন — সঠিক উত্তর পেয়েছেন?",
  ];
  tasks.forEach(function (t) { form.addMultipleChoiceItem().setTitle(t).setChoiceValues(yp).setRequired(true); });
  form.addTextItem().setTitle("O1. Which question did you ask in T4? / T4-এ কোন প্রশ্ন করেছিলেন?").setRequired(false);
  form.addMultipleChoiceItem()
    .setTitle("T5. Ask something NOT about MU (e.g., weather, cricket). Did the chatbot politely refuse? / MU-এর বাইরের প্রশ্ন করুন — চ্যাটবট কি ভদ্রভাবে উত্তর দিতে অস্বীকার করেছে?")
    .setChoiceValues(yp).setRequired(true);

  // ---------- opinion statements ----------
  form.addPageBreakItem().setTitle("Part 3 — Your opinion / আপনার মতামত")
    .setHelpText("1 = Strongly disagree / একদম একমত নই, 5 = Strongly agree / সম্পূর্ণ একমত");
  const likert = [
    "L1. Overall, I am satisfied with the chatbot. / সব মিলিয়ে আমি চ্যাটবট নিয়ে সন্তুষ্ট।",
    "L2. The answers were accurate. / উত্তরগুলো সঠিক ছিল।",
    "L3. The answers were relevant to my questions. / উত্তরগুলো আমার প্রশ্নের সাথে প্রাসঙ্গিক ছিল।",
    "L4. The answers were clear and easy to understand. / উত্তরগুলো পরিষ্কার ও সহজবোধ্য ছিল।",
    "L5. The chatbot responded quickly. / চ্যাটবট দ্রুত উত্তর দিয়েছে।",
    "L6. It understood my questions in Bangla. / বাংলায় করা প্রশ্ন বুঝতে পেরেছে।",
    "L7. It understood my questions in Banglish. / Banglish-এ করা প্রশ্ন বুঝতে পেরেছে।",
    "L8. Seeing the sources made me trust the answers more. / তথ্যের উৎস দেখে উত্তরের উপর আস্থা বেড়েছে।",
    "L9. When it did not know, it said so instead of guessing. / না জানলে আন্দাজ না করে জানি না বলেছে।",
    "L10. This is faster than visiting the office or searching the website. / অফিসে যাওয়া বা ওয়েবসাইট খোঁজার চেয়ে এটা দ্রুত।",
    "L11. 24/7 availability would be useful for me. / ২৪ ঘণ্টা চালু থাকলে আমার কাজে লাগবে।",
    "L12. I would recommend this chatbot to other MU students. / অন্য MU শিক্ষার্থীদের এটি ব্যবহার করতে বলব।",
  ];
  likert.forEach(function (q) {
    form.addScaleItem().setTitle(q).setBounds(1, 5).setLabels("Strongly disagree", "Strongly agree").setRequired(true);
  });

  // ---------- System Usability Scale (keep the standard English wording) ----------
  form.addPageBreakItem().setTitle("Part 4 — Ease of use (System Usability Scale)")
    .setHelpText("1 = Strongly disagree, 5 = Strongly agree");
  const sus = [
    "S1. I think that I would like to use this system frequently.",
    "S2. I found the system unnecessarily complex.",
    "S3. I thought the system was easy to use.",
    "S4. I think that I would need the support of a technical person to be able to use this system.",
    "S5. I found the various functions in this system were well integrated.",
    "S6. I thought there was too much inconsistency in this system.",
    "S7. I would imagine that most people would learn to use this system very quickly.",
    "S8. I found the system very cumbersome to use.",
    "S9. I felt very confident using the system.",
    "S10. I needed to learn a lot of things before I could get going with this system.",
  ];
  sus.forEach(function (q) {
    form.addScaleItem().setTitle(q).setBounds(1, 5).setLabels("Strongly disagree", "Strongly agree").setRequired(true);
  });

  // ---------- open questions ----------
  form.addPageBreakItem().setTitle("Part 5 — Comments / মন্তব্য");
  form.addParagraphTextItem().setTitle("O2. What did you like most? / কোন বিষয়টি সবচেয়ে ভালো লেগেছে?").setRequired(false);
  form.addParagraphTextItem().setTitle("O3. What problems did you face? Any suggestions? / কী সমস্যা হয়েছে? কোনো পরামর্শ?").setRequired(false);

  Logger.log("Edit link : " + form.getEditUrl());
  Logger.log("Share link: " + form.getPublishedUrl());
}
