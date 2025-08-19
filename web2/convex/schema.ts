import { defineSchema, defineTable } from 'convex/server';
import { v } from 'convex/values';

import { authTables } from '@convex-dev/auth/server';

const schema = defineSchema({
  ...authTables,
  studyUsers: defineTable({
    username: v.string(),
    password: v.string(),
  }),
  // Your other tables...
});

export default schema;
