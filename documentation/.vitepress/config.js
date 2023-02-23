export default {

    lang: 'en-US',

    title: 'Snow and Cloud Segmentation',
    description: 'Deep Learning for Accurate Snow and Cloud Segmentation in Alpine Landscapes',

    themeConfig: {

        sidebar: {
            '/docs/': [

                {
                    text: 'Data Sources and Pre-Processing',
                    items: [
                        {text: 'Getting Started', link: 'docs/getting-started'},
                        {text: 'Data Source', link: 'docs/data_sources'},
                    ]
                },

                {
                    text: 'Nice to Know',
                    items: [
                        {text: 'Python and Conda', link: 'docs/python_and_conda'},
                        {text: 'Working with Euler', link: 'docs/euler'},
                    ]
                },

                {
                    text: 'Models',
                    items: [
                        {text: 's2cloudless', link: 'docs/s2cloudless'},
                    ]
                }


            ]
        },

        nav: [
            {text: 'Start', link: 'index'},
            {text: 'Getting Started', link: 'docs/getting-started'}
        ]
    }
}
