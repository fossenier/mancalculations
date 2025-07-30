// import { v } from "convex/values";

// import { createAccount } from "@convex-dev/auth/server";

// import { mutation } from "../_generated/server";

// export const createStudyUsers = mutation({
//   args: {
//     studyId: v.string(),
//     count: v.number(),
//   },
//   handler: async (ctx, args) => {
//     await createAccount(ctx, {
//       provider: "test",
//       account: { id: args.studyId, secret: "password" },
//       profile: {},
//       shouldLinkViaEmail: false,
//       shouldLinkViaPhone: false,
//     });
//   },
// });
