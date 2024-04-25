import { convertJsonSchemaToGrammar } from '../grammar'

const schema = {
  oneOf: [
    {
      type: 'object',
      properties: {
        function: { const: 'create_event' },
        arguments: {
          type: 'object',
          properties: {
            title: { type: 'string' },
            date: { type: 'string' },
            time: { type: 'string' },
          },
        },
      },
      required: ['function', 'arguments'],
    },
    {
      type: 'object',
      properties: {
        function: { const: 'image_search' },
        arguments: {
          type: 'object',
          properties: {
            query: { type: 'string' },
          },
        },
      },
      required: ['function', 'arguments'],
    },
  ],
}

test('with prop order', () => {
  expect(
    convertJsonSchemaToGrammar({
      schema,
      propOrder: { function: 0, arguments: 1 },
    }),
  ).toMatchSnapshot()
})

test('without prop order', () => {
  expect(convertJsonSchemaToGrammar({ schema })).toMatchSnapshot()
})
